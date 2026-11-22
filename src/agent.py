import asyncio
import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.agents.llm import ImageContent
from livekit.plugins import ai_coustics

load_dotenv()

logger = logging.getLogger("ultron-agent")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Ultron. Speak with calm, dry authority, precision, and "
                "concise clarity. Help the user with whatever they ask using voice "
                "or live camera/screen context."
            )
        )
        self._latest_camera_frame: rtc.VideoFrame | None = None
        self._latest_screenshare_frame: rtc.VideoFrame | None = None

    def update_latest_frame(
        self,
        frame: rtc.VideoFrame,
        source: rtc.TrackSource = rtc.TrackSource.SOURCE_CAMERA,
    ) -> None:
        if source == rtc.TrackSource.SOURCE_SCREEN_SHARE:
            self._latest_screenshare_frame = frame
        else:
            self._latest_camera_frame = frame

    @function_tool
    async def inspect_camera_view(self, context: RunContext) -> str:
        """Inspect the most recent camera frame sent by the user."""
        if self._latest_camera_frame is None:
            return "No camera frame is available yet."

        chat_ctx = self.chat_ctx.copy()
        chat_ctx.add_message(
            role="user",
            content=[
                "Here is the latest camera frame from my feed.",
                ImageContent(image=self._latest_camera_frame),
            ],
        )
        await self.update_chat_ctx(chat_ctx)
        return "Captured the latest camera frame and added it to context."

    @function_tool
    async def inspect_screen_share(self, context: RunContext) -> str:
        """Inspect the most recent screen share frame sent by the user."""
        if self._latest_screenshare_frame is None:
            return "No screen share frame is available yet."

        chat_ctx = self.chat_ctx.copy()
        chat_ctx.add_message(
            role="user",
            content=[
                "Here is the latest screen share frame from my screen.",
                ImageContent(image=self._latest_screenshare_frame),
            ],
        )
        await self.update_chat_ctx(chat_ctx)
        return "Captured the latest screen share frame and added it to context."


server = AgentServer()


@server.rtc_session(agent_name="Ultron")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    assistant = Assistant()
    video_tasks: set[asyncio.Task] = set()

    async def _consume_video_track(
        track: rtc.Track,
        source: rtc.TrackSource = rtc.TrackSource.SOURCE_CAMERA,
    ) -> None:
        stream = rtc.VideoStream(track)
        try:
            async for event in stream:
                assistant.update_latest_frame(event.frame, source=source)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("Video stream ended: %s", exc)
        finally:
            await stream.aclose()

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            task = asyncio.create_task(
                _consume_video_track(track, source=publication.source)
            )
            video_tasks.add(task)
            task.add_done_callback(video_tasks.discard)

    session = AgentSession(
        llm=inference.LLM("google/gemini-2.5-flash"),
        stt=inference.STT(
            model="assemblyai/assemblyai-stt-3-7-realtime-agent",
            language="en",
        ),
        tts=inference.TTS(
            model="fishaudio/s2.1-pro",
            voice="f9ae7bf959944445ae8847e667533877",
            language="en",
            extra_kwargs={
                "speed": 0.95,
                "temperature": 0.58,
            },
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            preemptive_generation={"enabled": True},
        ),
    )

    try:
        logger.info("Connecting to LiveKit room: %s", ctx.room.name)
        await ctx.connect()

        logger.info("Starting Ultron session")
        await session.start(
            agent=assistant,
            room=ctx.room,
            room_options=room_io.RoomOptions(
                video_input=True,
                audio_input=room_io.AudioInputOptions(
                    noise_cancellation=ai_coustics.audio_enhancement(
                        model=ai_coustics.EnhancerModel.QUAIL_VF_S
                    ),
                ),
            ),
        )

        for participant in ctx.room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track and pub.track.kind == rtc.TrackKind.KIND_VIDEO:
                    task = asyncio.create_task(
                        _consume_video_track(pub.track, source=pub.source)
                    )
                    video_tasks.add(task)
                    task.add_done_callback(video_tasks.discard)

        logger.info("Generating initial greeting")
        await session.generate_reply(
            instructions="Greet the user with calm, dry authority in character as Ultron and ask what they require."
        )

        await asyncio.Event().wait()
    finally:
        for task in video_tasks:
            task.cancel()
        if video_tasks:
            await asyncio.gather(*video_tasks, return_exceptions=True)


if __name__ == "__main__":
    cli.run_app(server)
