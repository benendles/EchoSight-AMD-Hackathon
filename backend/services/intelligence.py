"""Optional cloud enrichment with local policy owning every navigation decision."""
from __future__ import annotations

import asyncio
import logging
import json
import re
import time

from .amd_vision import AMDVisionClient
from .cloud_http import CloudError, seconds
from .evolus import EvolusClient
from ..sessions import Observation

logger = logging.getLogger(__name__)


def navigation_question(question: str) -> bool:
    return bool(re.search(r"\b(safe|cross|crossing|walk|walking|move|navigate|navigation|path|route|which way|where should i go|where do i go|direction should)\b", question.lower()))


def crossing_question(question: str) -> bool:
    return bool(re.search(r"\b(cross|crossing)\b", question.lower()))


def needs_amd(question: str, scene: list[dict], *, upload: bool = False) -> bool:
    q = question.lower()
    if upload or any(term in q for term in ("detail", "everything", "surroundings", "explain", "image", "photo", "map", "floorplan", "read", "behind", "color", "colour")):
        return True
    return not scene or not any(str(obj.get("class", "")).lower() in q for obj in scene) and not any(term in q for term in ("ahead", "left", "right", "people", "objects"))


def local_answer(question: str, observation: Observation) -> str:
    q = question.lower()
    objects = [x for x in observation.scene if x.get("confidence", 0) >= .45]
    if crossing_question(q):
        return "I can't reliably determine whether it's safe to cross from this camera view."
    if navigation_question(q):
        if observation.source == "upload":
            return "This is an uploaded image, not a live camera view. I can't assess a current walking route from it."
        return observation.navigation["message"]
    direction = next((x for x in ("left", "right") if x in q), "center" if "ahead" in q or "in front" in q else None)
    if direction:
        objects = [x for x in objects if x.get("direction") == direction]
    if "how many" in q:
        if "people" in q or "person" in q:
            count = sum(x.get("class") == "person" for x in objects)
            return f"I detect {count} {'person' if count == 1 else 'people'} in this view."
        return f"I detect {len(objects)} objects in this view."
    # Narrow a question to an explicitly named detected object.
    named = [x for x in objects if str(x.get("class", "")) in q or (x.get("class") == "person" and "people" in q)]
    existence = re.search(r"\b(?:is there|do you see|can you see) (?:a |an |any )?(chair|bench|bus|car|person|table|door|stairs)\b", q)
    if existence and not named:
        return f"I don't detect a {existence.group(1)} in this view. It may be outside the view or unrecognized."
    if named and "behind" not in q:
        objects = named
    if not objects:
        return "I don't detect recognizable objects there. That does not establish that the area is clear."
    parts = []
    for obj in objects[:3]:
        meters = obj.get("distance_meters")
        distance = f", approximately {meters:.1f} meters away" if isinstance(meters, (int, float)) and meters > 0 else f", {obj.get('distance', 'unknown')} relative distance"
        parts.append(f"a {obj['class']} {obj.get('position', 'in view')}{distance}")
    if "behind" in q:
        return "I can't establish what is behind an occluded object from detections alone. I detect " + "; ".join(parts) + "."
    return "I detect " + "; ".join(parts) + "."


def unsafe_cloud_answer(answer: str) -> bool:
    return bool(re.search(r"\b(safe|safely|(?:path|road|route) is clear|go ahead|move (left|right|forward)|walk (forward|ahead)|cross now)\b", answer.lower()))


def unsupported_cloud_claim(answer: str, scene: list[dict], amd: dict | None) -> bool:
    """Additional conservative checks; reference IDs alone do not prove grounding."""
    evidence = json.dumps(scene).lower() + ' ' + ' '.join((amd or {}).get('observations', [])).lower()
    # Critical environmental features must occur in the supplied evidence.
    for pattern in (r'\bdoors?|doorways?\b', r'\bstairs?|staircases?\b', r'\bcurbs?|kerbs?\b',
                    r'\bhandrails?\b', r'\bcrossings?|crosswalks?\b', r'\btraffic (signals?|lights?)\b'):
        if re.search(pattern, answer.lower()) and not re.search(pattern, evidence):
            return True
    # Cloud vision has no metric calibration. Reject invented meter values even
    # if the agent supplies an otherwise valid local object reference.
    distances = [float(x['distance_meters']) for x in scene if isinstance(x.get('distance_meters'), (int, float)) and x['distance_meters'] > 0]
    if re.search(r'\b(?:meters?|metres?)\b', answer.lower()) and not distances:
        return True
    words = dict(zip(('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'), range(1, 11)))
    for token in re.findall(r'\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:meters?|metres?)\b', answer.lower()):
        value = words[token] if token in words else float(token)
        if not any(abs(value - distance) <= .6 for distance in distances):
            return True
    return False


class Intelligence:
    def __init__(self, amd: AMDVisionClient | None = None, evolus: EvolusClient | None = None):
        self.amd, self.evolus = amd, evolus
        self.limit = asyncio.Semaphore(4)

    @classmethod
    def from_environment(cls) -> Intelligence:
        clients = []
        for factory in (AMDVisionClient.from_environment, EvolusClient.from_environment):
            try:
                clients.append(factory())
            except (CloudError, ValueError):
                logger.warning("cloud_configuration_invalid service=%s", factory.__qualname__)
                clients.append(None)
        return cls(*clients)

    async def close(self) -> None:
        for client in (self.amd, self.evolus):
            if client:
                await client.close()

    async def enrich(self, question: str, observation: Observation, *, upload: bool = False) -> dict:
        result = {"answer": local_answer(question, observation), "amd": None,
                  "sources": ["local"], "cloud": {"amd": "not_requested", "evolus": "not_requested"}}
        if navigation_question(question):
            return result
        started = time.monotonic()
        # One total budget includes queue wait, image preparation, verification,
        # AMD and Evolus. Cancellation is allowed to propagate to HTTPX.
        try:
            async with asyncio.timeout(seconds("ECHOSIGHT_CLOUD_TOTAL_TIMEOUT_SECONDS", 30, 30)):
                async with self.limit:
                    if needs_amd(question, observation.scene, upload=upload):
                        result["cloud"]["amd"] = "disabled"
                        if self.amd:
                            try:
                                result["cloud"]["amd"] = "pending"
                                result["amd"] = await self.amd.analyze(observation.frame, question)
                                result["sources"].append("amd")
                                result["cloud"]["amd"] = "success"
                            except CloudError as exc:
                                result["cloud"]["amd"] = str(exc)
                    result["cloud"]["evolus"] = "disabled"
                    if self.evolus:
                        try:
                            result["cloud"]["evolus"] = "pending"
                            reply = await self.evolus.answer(question, observation.scene, observation.navigation,
                                                             result["amd"], source=observation.source, age_seconds=observation.age)
                            if unsafe_cloud_answer(reply["answer"]) or unsupported_cloud_claim(reply["answer"], observation.scene, result["amd"]):
                                raise CloudError("unsafe_answer_rejected")
                            result["answer"] = reply["answer"]
                            result["evidence"] = reply["evidence"]
                            result["sources"].append("evolus")
                            result["cloud"]["evolus"] = "success"
                        except CloudError as exc:
                            result["cloud"]["evolus"] = str(exc)
        except TimeoutError:
            result["cloud"]["deadline"] = "timeout"
            for provider in ("amd", "evolus"):
                if result["cloud"].get(provider) == "pending":
                    result["cloud"][provider] = "timeout"
        result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
        # AMD success is visible as evidence but is never silently promoted to
        # an agent answer if Evolus failed. Local descriptions remain the fallback.
        return result
