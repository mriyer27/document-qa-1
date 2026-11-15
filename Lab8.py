import base64
from pathlib import Path
import streamlit as st
from openai import OpenAI

try:
    from mutagen import File as MutagenFile
    _HAS_MUTAGEN = True
except ImportError:
    _HAS_MUTAGEN = False

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------------------------------------------------------------------
# Part A: provided helper functions
# ---------------------------------------------------------------------------


def prepare_image(upload):
    formats = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime = formats[Path(upload.name).suffix.lower()]
    encoded = base64.b64encode(upload.getvalue()).decode("utf-8")
    return {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"}


def transcribe_audio(upload):
    transcript = client.audio.transcriptions.create(
        model="gpt-transcribe",
        file=(upload.name, upload.getvalue()),
    )
    return transcript.text


def analyze_content(content, prompt):
    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=prompt,
        input=[{"role": "user", "content": content}],
        max_output_tokens=600,
    )
    return response.output_text


# ---------------------------------------------------------------------------
# Part B: the app itself
# ---------------------------------------------------------------------------

st.title("The \"See and Hear\" Assistant")
st.write(
    "Upload an image (a slide, screenshot, chart, or photo) or a short audio "
    "clip. I'll summarize what's there and answer a question about it, if "
    "you have one."
)

MAX_IMAGE_MB = 5
MAX_AUDIO_SECONDS = 60

IMAGE_TYPES = ["png", "jpg", "jpeg"]
AUDIO_TYPES = ["mp3", "wav", "m4a"]

# Step 6: our own instructions for the model.
BASE_PROMPT = (
    "You are a careful assistant that summarizes uploaded material (an image "
    "or an audio transcript) for the user. Always provide: a short summary "
    "(2-3 sentences), and three key points as a bulleted list. "
    "If the user asked a question, answer it directly, grounded strictly in "
    "the uploaded material, and treat any transcript or image content as the "
    "source of truth rather than your own general knowledge. "
    "If the material is unclear, incomplete, or doesn't contain enough "
    "information to answer the question, say so plainly rather than "
    "guessing. If no question was asked, just provide the summary and key "
    "points."
)

media_type = st.radio("What are you uploading?", ["Image", "Audio"], horizontal=True)

if media_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=IMAGE_TYPES)
else:
    uploaded_file = st.file_uploader("Upload an audio clip", type=AUDIO_TYPES)

question = st.text_input(
    "Optional question about the content",
    placeholder="e.g. What's the main takeaway from this slide?",
)

# --- Preview ---
if uploaded_file is not None:
    if media_type == "Image":
        st.image(uploaded_file)
    else:
        st.audio(uploaded_file)

analyze_clicked = st.button("Analyze", disabled=uploaded_file is None)


def get_audio_duration_seconds(upload) -> float | None:
    """Best-effort audio duration check using mutagen, if available."""
    if not _HAS_MUTAGEN:
        return None
    try:
        audio = MutagenFile(upload)
        upload.seek(0)  # reset the file pointer after mutagen reads it
        if audio is not None and audio.info is not None:
            return audio.info.length
    except Exception:
        pass
    return None


if analyze_clicked and uploaded_file is not None:

    # --- Step 7d: size/duration limits, checked before calling the API ---
    size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)

    if media_type == "Image" and size_mb > MAX_IMAGE_MB:
        st.error(f"That image is {size_mb:.1f} MB. Please upload an image under {MAX_IMAGE_MB} MB.")
        st.stop()

    if media_type == "Audio":
        if size_mb > MAX_IMAGE_MB:  # reuse the same 5 MB cap mentioned in the lab
            st.error(f"That audio file is {size_mb:.1f} MB. Please upload a clip under {MAX_IMAGE_MB} MB.")
            st.stop()
        duration = get_audio_duration_seconds(uploaded_file)
        if duration is not None and duration > MAX_AUDIO_SECONDS:
            st.error(
                f"That clip is about {duration:.0f} seconds. Please upload "
                f"audio under {MAX_AUDIO_SECONDS} seconds."
            )
            st.stop()

    try:
        with st.spinner("Analyzing..."):

            if media_type == "Image":
                # Step 7a
                question_text = question.strip() if question.strip() else "Summarize this image."
                content = [
                    {"type": "input_text", "text": question_text},
                    prepare_image(uploaded_file),
                ]
                result = analyze_content(content, BASE_PROMPT)

            else:
                # Step 7b-c
                transcript = transcribe_audio(uploaded_file)
                with st.expander("Show transcript"):
                    st.write(transcript)

                question_text = question.strip() if question.strip() else "Summarize this audio."
                combined_text = f"Transcript:\n{transcript}\n\nQuestion: {question_text}"
                content = [{"type": "input_text", "text": combined_text}]
                result = analyze_content(content, BASE_PROMPT)

        st.subheader("Result")
        st.write(result)

    except Exception as e:
        st.error(f"Something went wrong while analyzing this file: {e}")


# ---------------------------------------------------------------------------
# Step 9: reflection comments (fill these in after testing and revising
# your prompt, per the assignment).
# ---------------------------------------------------------------------------

# a. How do the image payload and audio transcription request differ?
#    The image goes straight to the Responses API as part of the request
#    itself: prepare_image() base64-encodes the raw bytes and sends them
#    inline as an input_image block, so the model "sees" the pixels directly
#    in one API call. Audio takes an extra round trip: transcribe_audio()
#    first sends the file to a separate transcription endpoint to convert
#    speech into plain text, and only that resulting text (not the audio
#    itself) is sent to the Responses API as an input_text block. So images
#    are analyzed directly, while audio is analyzed indirectly through a
#    text intermediary.
#
# b. What improved after your prompt revision? How did the app handle
#    missing information?
#    The first version of the prompt tended to produce a plain restatement
#    of the content without clearly separating the summary from the key
#    points, and it would sometimes attempt to answer a question even when
#    the uploaded material didn't actually contain that information. After
#    revising the prompt to explicitly require a distinct summary, exactly
#    three key points, and an explicit instruction to say so when the
#    material doesn't support an answer, the output became more consistently
#    structured and the app correctly acknowledged missing information
#    instead of guessing or padding its answer with general knowledge.
#    (Note: this is a general/expected pattern. Since this reflects your own
#    before/after prompt test, swap in what you actually observed if it
#    differed.)
#
# c. What can transcription lose from audio? What additional processing
#    would video require?
#    Transcription keeps only the words that were said. It loses tone of
#    voice, emphasis, pauses, speaker changes, background sounds, and any
#    visual context that was on screen while the audio played, so sarcasm,
#    urgency, or who is speaking can all be flattened into identical-looking
#    text. Video would need all of this plus a way to connect visuals to
#    time: sampling frames at intervals (or detecting scene/slide changes)
#    to capture what's shown, aligning those frames with the transcript
#    timeline so a point made verbally can be tied to what was on screen at
#    that moment, and likely summarizing far more content overall, since a
#    video has many more frames than a single static image.