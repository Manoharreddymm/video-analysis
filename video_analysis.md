`    `**Complete Video Analysis Using Streamlit framework** 

This framework contains mix up with multiple concepts like transformers piplines, large language models and multi model llms.

By using the streamlit framework we developed complete video analysis with addition to that we use libraries like-

- Pydub
- Openai
- Transformers
- Cv2
- Io 

**Flow of video analysis:**

- Taking an input from the user by using the help of streamlit file\_uploader then displaying the video.
- To print the transcription of the audio we need to first extract the audio from the video. To extract the audio from the video is saved as .wav file this contains the audio of the recording.
- Using the asr\_pipeline we are going to convert our audio into transcription then displaying the audio text.
- Following with that text as input we are checking the sentiment of the speakers by sentimeny-analysis.
- Summarization pipeline is used to summarize the transcription of the audio in the video.
- By using openai we are going to create question and answering for the video. we are passing the transcription and question as input to the openai and getting the answers related to it.

Adding a sidebar to the framework displaying the frames of the video for every 15 seconds using opencv. Then describing the frame using a multimodel llm BLIP.

To conclude, this framework gives the brief understanding of a video which will capture every point in the video.    

