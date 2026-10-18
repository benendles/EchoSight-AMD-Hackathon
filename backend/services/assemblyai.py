import asyncio
import base64
import json
import os
import ssl
from pathlib import Path

import certifi
import pyaudio
import websockets
from dotenv import load_dotenv


# CONFIGURATION

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

ASSEMBLYAI_WS_URL = "wss://agents.assemblyai.com/v1/ws"

# AssemblyAI Voice Agent audio format
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_FORMAT = pyaudio.paInt16

# 1200 frames = 50 ms at 24 kHz
FRAMES_PER_BUFFER = 1200


# Microphone protection

# Keep microphone disabled for this long after EchoSight finishes
# speaking. This prevents the microphone from hearing the speaker.
MIC_COOLDOWN_SECONDS = 5.0

# Number of microphone chunks to discard after the cooldown.
# 10 chunks × 50 ms = 500 ms of buffered microphone audio.
MIC_FLUSH_CHUNKS = 10

# Use certifi's CA bundle so Python can verify AssemblyAI's TLS
# certificate.
SSL_CONTEXT = ssl.create_default_context(
    cafile=certifi.where()
)


# ECHOSIGHT TOOL

ECHOSIGHT_TOOL = {
    "type": "function",
    "name": "analyze_echo_sight_scene",
    "description": (
        "Analyze the user's current camera scene using EchoSight "
        "computer vision. Use this tool when the user asks what "
        "they see, what is ahead, about obstacles, navigation, "
        "whether the path is safe, or what they should do next."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": (
                    "The user's spoken request about the "
                    "current camera scene."
                ),
            }
        },
        "required": ["request"],
    },
}


# VOICE AGENT SYSTEM PROMPT

SYSTEM_PROMPT = """
You are EchoSight, a voice assistant designed to help blind or
visually impaired users understand their surroundings.

You have access to the analyze_echo_sight_scene tool.

Use this tool whenever the user's question depends on what the
camera currently sees.

Examples include:

- What do you see?
- What's ahead?
- Is there anything in front of me?
- Is the path clear?
- Can I walk forward?
- Where should I go?
- Is there an obstacle?
- What's on my left?
- What's on my right?
- How many people are there?
- How many objects do you see?

Do not invent visual information.

Keep responses short, clear, natural, and easy to understand
when spoken.

If the vision system gives a navigation instruction, communicate
that instruction clearly and prioritize it.

EchoSight is an assistive prototype and is not a safety-certified
collision-avoidance system.
""".strip()


# AUDIO INPUT / OUTPUT

class AudioIO:
    """
    Handles microphone input and speaker output using PyAudio.
    """

    def __init__(self):
        self.audio = pyaudio.PyAudio()

        self.microphone = None
        self.speaker = None

    def open(self):
        """
        Open microphone and speaker streams.
        """

        self.microphone = self.audio.open(
            format=SAMPLE_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
        )

        self.speaker = self.audio.open(
            format=SAMPLE_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
        )

        print(
            "[Audio] Microphone and speaker ready."
        )

    def read_microphone(self):
        """
        Read one chunk of microphone audio.
        """

        return self.microphone.read(
            FRAMES_PER_BUFFER,
            exception_on_overflow=False,
        )

    def discard_microphone_audio(self, chunks=MIC_FLUSH_CHUNKS):
        """
        Discard buffered microphone audio.

        This is important after EchoSight speaks because some
        microphone audio may still contain the tail end of the
        speaker output.
        """

        for _ in range(chunks):

            try:
                self.microphone.read(
                    FRAMES_PER_BUFFER,
                    exception_on_overflow=False,
                )

            except Exception:
                break

    def play_audio(self, audio_bytes):
        """
        Play PCM audio received from AssemblyAI.
        """

        self.speaker.write(audio_bytes)

    def close(self):
        """
        Close audio devices.
        """

        if self.microphone is not None:
            self.microphone.stop_stream()
            self.microphone.close()

        if self.speaker is not None:
            self.speaker.stop_stream()
            self.speaker.close()

        self.audio.terminate()

        print(
            "[Audio] Audio devices closed."
        )


# SEND MICROPHONE AUDIO

async def send_microphone(ws, audio_io, listening_event):
    """
    Continuously send microphone audio to AssemblyAI.

    The listening_event controls whether the microphone is
    currently allowed to send audio.

    When EchoSight speaks, the event is cleared and no microphone
    audio is sent.

    When the cooldown finishes, the event is set again.
    """

    try:

        while True:

            # ------------------------------------------------
            # Wait until microphone is enabled.
            # ------------------------------------------------

            await listening_event.wait()

            # ------------------------------------------------
            # Read microphone audio in a background thread.
            # ------------------------------------------------

            audio_bytes = await asyncio.to_thread(
                audio_io.read_microphone
            )

            # ------------------------------------------------
            # Check AGAIN after reading.
            #
            # This prevents a microphone chunk captured at the
            # exact moment EchoSight starts speaking from being
            # sent to AssemblyAI.
            # ------------------------------------------------

            if not listening_event.is_set():
                continue

            encoded_audio = base64.b64encode(
                audio_bytes
            ).decode("ascii")

            message = {
                "type": "input.audio",
                "audio": encoded_audio,
            }

            await ws.send(
                json.dumps(message)
            )

    except asyncio.CancelledError:

        pass

    except websockets.ConnectionClosed:

        pass

    except Exception as error:

        print(
            f"[Audio] Microphone error: {error}"
        )


# ============================================================
# RECEIVE ASSEMBLYAI EVENTS
# ============================================================

async def receive_messages(
    ws,
    audio_io,
    on_command,
    listening_event,
):
    """
    Receive and process AssemblyAI Voice Agent events.
    """

    async for raw_message in ws:

        message = json.loads(raw_message)

        message_type = message.get("type")

        # ----------------------------------------------------
        # SESSION READY
        # ----------------------------------------------------

        if message_type == "session.ready":

            listening_event.set()

            print(
                "[AssemblyAI] Session ready."
            )

            print(
                "[EchoSight] Ready. Microphone active."
            )

        # ----------------------------------------------------
        # USER TRANSCRIPT
        # ----------------------------------------------------

        elif message_type == "transcript.user":

            text = message.get(
                "text",
                ""
            ).strip()

            if text:

                print(
                    f"[You] {text}"
                )

        # ----------------------------------------------------
        # TOOL CALL
        # ----------------------------------------------------

        elif message_type == "tool.call":

            # ------------------------------------------------
            # Disable microphone while computer vision runs.
            # ------------------------------------------------

            listening_event.clear()

            print(
                "[EchoSight] Microphone inactive "
                "while analyzing scene..."
            )

            tool_call_id = (
                message.get("call_id")
                or message.get("tool_call_id")
                or message.get("id")
            )

            arguments = message.get(
                "arguments",
                {}
            )

            # Sometimes arguments arrive as JSON text.
            if isinstance(arguments, str):

                try:

                    arguments = json.loads(
                        arguments
                    )

                except json.JSONDecodeError:

                    arguments = {
                        "request": arguments
                    }

            request = str(
                arguments.get(
                    "request",
                    ""
                )
            ).strip()

            print(
                f"[EchoSight] Vision request: {request}"
            )

            try:

                # Call our existing EchoSight vision pipeline.
                result = await asyncio.to_thread(
                    on_command,
                    request
                )

                if not isinstance(result, str):

                    result = json.dumps(
                        result
                    )

            except Exception as error:

                print(
                    f"[EchoSight] Tool error: {error}"
                )

                result = (
                    "I could not analyze "
                    "the scene right now."
                )

            # ------------------------------------------------
            # Send vision result back to AssemblyAI.
            # ------------------------------------------------

            await ws.send(
                json.dumps(
                    {
                        "type": "tool.result",
                        "call_id": tool_call_id,
                        "result": result,
                    }
                )
            )

        # ----------------------------------------------------
        # AI AUDIO
        # ----------------------------------------------------

        elif message_type == "reply.audio":

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Immediately disable microphone as soon as
            # AssemblyAI starts speaking.
            # ------------------------------------------------

            if listening_event.is_set():

                listening_event.clear()

                print(
                    "[EchoSight] Speaking... "
                    "Microphone inactive."
                )

            # Current AssemblyAI event uses "data".
            # "audio" is kept as a fallback.
            audio_b64 = (
                message.get("data")
                or message.get("audio")
            )

            if audio_b64:

                try:

                    audio_bytes = base64.b64decode(
                        audio_b64
                    )

                    # Speaker output is blocking.
                    await asyncio.to_thread(
                        audio_io.play_audio,
                        audio_bytes
                    )

                except Exception as error:

                    print(
                        f"[Audio] Playback error: {error}"
                    )

        # ----------------------------------------------------
        # AI TEXT
        # ----------------------------------------------------

        elif message_type in (
            "transcript.agent",
            "reply.transcript"
        ):

            text = message.get(
                "text",
                ""
            ).strip()

            if text:

                print(
                    f"[EchoSight] {text}"
                )

        # ----------------------------------------------------
        # RESPONSE COMPLETE
        # ----------------------------------------------------

        elif message_type == "reply.done":

            # ------------------------------------------------
            # Keep microphone disabled.
            # ------------------------------------------------

            listening_event.clear()

            print(
                "[EchoSight] Response finished."
            )

            print(
                "[EchoSight] Microphone inactive "
                f"for {MIC_COOLDOWN_SECONDS:.0f} seconds..."
            )

            # ------------------------------------------------
            # Wait before allowing the microphone to listen.
            # ------------------------------------------------

            await asyncio.sleep(
                MIC_COOLDOWN_SECONDS
            )

            # ------------------------------------------------
            # Flush any microphone audio that accumulated
            # while the microphone was disabled.
            # ------------------------------------------------

            print(
                "[EchoSight] Clearing buffered "
                "microphone audio..."
            )

            await asyncio.to_thread(
                audio_io.discard_microphone_audio,
                MIC_FLUSH_CHUNKS
            )

            # ------------------------------------------------
            # Re-enable microphone.
            # ------------------------------------------------

            listening_event.set()

            print(
                "[EchoSight] Ready. Microphone active."
            )

        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        elif message_type in (
            "session.error",
            "error"
        ):

            print(
                f"[AssemblyAI] Error: {message}"
            )

            # Keep microphone disabled if the session reports
            # an error.
            listening_event.clear()

        # ----------------------------------------------------
        # SESSION ENDED
        # ----------------------------------------------------

        elif message_type == "session.ended":

            listening_event.clear()

            print(
                "[AssemblyAI] Session ended."
            )

            break


# ============================================================
# START ASSEMBLYAI VOICE AGENT
# ============================================================

async def _run_voice_agent(on_command):
    """
    Connect EchoSight to the AssemblyAI Voice Agent API.
    """

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if not ASSEMBLYAI_API_KEY:

        raise RuntimeError(
            f"ASSEMBLYAI_API_KEY was not found in:\n"
            f"{ENV_PATH}"
        )

    print(
        "[AssemblyAI] API key loaded."
    )

    # --------------------------------------------------------
    # INITIALIZE AUDIO
    # --------------------------------------------------------

    audio_io = AudioIO()

    audio_io.open()

    # --------------------------------------------------------
    # Microphone state.
    #
    # SET   = microphone can send audio
    # CLEAR = microphone is temporarily disabled
    # --------------------------------------------------------

    listening_event = asyncio.Event()

    try:

        # ----------------------------------------------------
        # CONNECT TO ASSEMBLYAI
        # ----------------------------------------------------

        print(
            "[AssemblyAI] Connecting..."
        )

        async with websockets.connect(
            ASSEMBLYAI_WS_URL,
            additional_headers={
                "Authorization": (
                    f"Bearer {ASSEMBLYAI_API_KEY}"
                )
            },
            max_size=None,
            ssl=SSL_CONTEXT,
        ) as ws:

            print(
                "[AssemblyAI] WebSocket connected."
            )

            # ------------------------------------------------
            # CONFIGURE VOICE AGENT
            # ------------------------------------------------

            session_update = {
                "type": "session.update",
                "session": {
                    "system_prompt": SYSTEM_PROMPT,
                    "tools": [
                        ECHOSIGHT_TOOL
                    ],
                    "output": {
                        "voice": "ivy"
                    },
                },
            }

            await ws.send(
                json.dumps(
                    session_update
                )
            )

            print(
                "[AssemblyAI] Waiting for session..."
            )

            # ------------------------------------------------
            # WAIT FOR SESSION READY
            # ------------------------------------------------

            while True:

                raw_message = await ws.recv()

                message = json.loads(
                    raw_message
                )

                message_type = message.get(
                    "type"
                )

                if message_type == "session.ready":

                    print(
                        "[AssemblyAI] "
                        "Session ready."
                    )

                    # Microphone can now start.
                    listening_event.set()

                    print(
                        "[EchoSight] "
                        "Ready. Microphone active."
                    )

                    break

                if message_type in (
                    "session.error",
                    "error"
                ):

                    raise RuntimeError(
                        "AssemblyAI error: "
                        f"{message}"
                    )

            # ------------------------------------------------
            # START MICROPHONE STREAM
            # ------------------------------------------------

            microphone_task = asyncio.create_task(
                send_microphone(
                    ws,
                    audio_io,
                    listening_event,
                )
            )

            try:

                await receive_messages(
                    ws,
                    audio_io,
                    on_command,
                    listening_event,
                )

            finally:

                microphone_task.cancel()

                await asyncio.gather(
                    microphone_task,
                    return_exceptions=True
                )

    finally:

        audio_io.close()


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_voice_agent(on_command):
    """
    Start the EchoSight realtime voice agent.

    on_command:
        Function that receives the user's spoken request
        and returns the EchoSight vision response.
    """

    try:

        asyncio.run(
            _run_voice_agent(
                on_command
            )
        )

    except KeyboardInterrupt:

        print(
            "\n[EchoSight] "
            "Voice agent stopped."
        )