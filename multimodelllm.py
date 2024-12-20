import streamlit as st
from pydub import AudioSegment
import tempfile
import os
import openai  
from transformers import pipeline, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
import cv2
import io

openai.api_key = ""  # Replace with your OpenAI API key

# Initialize pipelines
asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
sentiment_pipeline = pipeline("sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
summarization_pipeline = pipeline("summarization")

# Initialize BLIP model and processor for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def extract_audio_from_video(video_path):
    audio = AudioSegment.from_file(video_path, format="mp4")
    temp_audio_path = tempfile.mktemp(suffix=".wav")
    audio.export(temp_audio_path, format="wav")
    return temp_audio_path

def transcribe_audio(audio_path):
    result = asr_pipeline(audio_path)
    transcription = result['text']
    return transcription

def split_text(text, max_length=512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def analyze_sentiment(text):
    chunks = split_text(text)
    results = []
    for chunk in chunks:
        result = sentiment_pipeline(chunk)
        results.append(result)
    
    positive_scores = [s[0]['score'] for s in results if s[0]['label'] == 'POSITIVE']
    negative_scores = [s[0]['score'] for s in results if s[0]['label'] == 'NEGATIVE']
    
    avg_positive_score = sum(positive_scores) / len(positive_scores) if positive_scores else 0
    avg_negative_score = sum(negative_scores) / len(negative_scores) if negative_scores else 0
    
    if avg_positive_score > avg_negative_score:
        overall_sentiment = 'POSITIVE'
        overall_score = avg_positive_score
    else:
        overall_sentiment = 'NEGATIVE'
        overall_score = avg_negative_score
    
    return overall_sentiment, overall_score

def summarize_text(text, max_chunk_size=1024):
    chunks = split_text(text, max_length=max_chunk_size)
    summaries = []
    for chunk in chunks:
        try:
            summary = summarization_pipeline(chunk, max_length=150, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            st.write(f"Error summarizing chunk: {e}")
            summaries.append("Error in summarization")
    return " ".join(summaries)

def gpt_question_answer(question, passage):
    prompt = f"Passage: {passage}\n\nQuestion: {question}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7,
    )
    answer = response['choices'][0]['message']['content'].strip()
    return answer

def describe_frame(frame):
    inputs = processor(images=frame, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Streamlit UI
st.title("Complete Video Analysis")

# Upload video file
uploaded_file = st.file_uploader("Upload a Video File", type=["mp4"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
        temp_video_path = temp_video_file.name
        temp_video_file.write(uploaded_file.read())
        
    st.video(temp_video_path)  # Display video
    
    # Extract audio from the video
    audio_path = extract_audio_from_video(temp_video_path)
    
    st.write("Transcribing audio...")
    transcription = transcribe_audio(audio_path)
    
    st.markdown("<h1 style='font-size: 36px;'>Transcription:</h1>", unsafe_allow_html=True)
    st.text_area("Transcript", transcription, height=300)
    
    # Perform sentiment analysis
    sentiment, score = analyze_sentiment(transcription)
    st.markdown("<h1 style='font-size: 36px;'>Sentiment Analysis:</h1>", unsafe_allow_html=True)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence Score: {score:.2f}")
    
    # Summarization button
    if st.button("Summarize Text"):
        summary = summarize_text(transcription)
        st.markdown("<h1 style='font-size: 36px;'>Summary</h1>", unsafe_allow_html=True)
        st.write(summary)
    
    # Question answering
    question = st.text_input("Ask a question based on the transcription:")
    if question:
        answer = gpt_question_answer(question, transcription)
        st.write("Answer:")
        st.write(answer)
    
    # Show frames and their descriptions in the sidebar
    st.sidebar.title("Video Frames and Descriptions")
    
    cap = cv2.VideoCapture(temp_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 15)  # Get frames every 15 seconds
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            description = describe_frame(frame)
            st.sidebar.image(frame, caption=f"Frame {frame_count // fps} seconds: {description}", use_column_width=True)
        frame_count += 1
    
    cap.release()

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(audio_path)
