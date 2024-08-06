import os
import pickle
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document
from flask import Flask, render_template, request, jsonify, send_file,current_app
from transformers import (
    T5ForConditionalGeneration, AutoTokenizer, AutoModelWithLMHead, 
    pipeline, CLIPProcessor, CLIPModel, T5Tokenizer, MT5ForConditionalGeneration, 
    Text2TextGenerationPipeline
)
from langchain.prompts import PromptTemplate
from PIL import Image
import torch
import numpy as np
import csv
import re
import base64
import speech_recognition as sr
from gtts import gTTS
import io
from lingua import Language, LanguageDetectorBuilder
import whisper
from pydub import AudioSegment
from flask_cors import CORS
import tempfile
import logging
import asyncio
from functools import partial
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.exc import ProgrammingError
from sqlalchemy import create_engine, text
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PICKLE_FILE = 'app_resources.pkl'
IMAGE_PICKLE_FILE = 'image.pkl'
# Check for required environment variables
required_env_vars = ['PORT', 'GOOGLE_API_KEY','DATABASE_URL']  # Add all required env vars
for var in required_env_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Required environment variable {var} is not set.")

STATIC_URL_PATH = "/static"
app = Flask(__name__, static_url_path=STATIC_URL_PATH)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
CORS(app, resources={r"/*": {"origins": os.getenv('ALLOWED_ORIGINS', '*').split(',')}})
db = SQLAlchemy(app)
UPLOAD_FOLDER = './data/images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class ZhGlossary(db.Model):
    __tablename__ = 'ZhGlossary'

    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String)
    target = db.Column(db.String)
    
class FAQ(db.Model):
    __tablename__ = 'FAQ'

    id = db.Column(db.Integer, primary_key=True)
    Question = db.Column(db.String)
    Answer = db.Column(db.String)

class BmGlossary(db.Model):
    __tablename__ = 'BmGlossary'

    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String)
    target = db.Column(db.String)

class ImageData(db.Model):
    __tablename__ = 'ImageData'

    id = db.Column(db.Integer, primary_key=True)
    Image_Path = db.Column(db.String)
    Summary = db.Column(db.String)
    
def create_database():
    engine = create_engine('postgresql://postgres:123oks321@localhost/postgres')
    conn = engine.connect()
    conn.execute(text("COMMIT"))
    try:
        conn.execute(text("CREATE DATABASE testdb"))
        logger.info("Database 'testdb' created successfully.")
    except ProgrammingError:
        logger.info("Database 'testdb' already exists.")
    finally:
        conn.close()
        
def init_db():
    
    try:
        create_database()
        with app.app_context():
            db.create_all()
        logger.info("Tables created successfully.")
    except SQLAlchemyError as e:
        logger.error(f"An error occurred initializing the database: {e}")
        

def save_resources(resources: Dict[str, Any],csv_resources_to_pickle: Dict[str,Any]) -> None:
    """Save picklable resources to a file."""
    resources_to_pickle = {k: v for k, v in resources.items() if k not in ['CLIP_MODEL','CLIP_PROCESSOR','IMAGE_DATA','IMAGE_ARR']}
    try:
        with open(PICKLE_FILE, 'wb') as f:
            pickle.dump(resources_to_pickle, f,protocol=pickle.HIGHEST_PROTOCOL)
        with open(IMAGE_PICKLE_FILE, 'wb') as file:
            pickle.dump({
                "IMAGE_ARR":resources["IMAGE_ARR"],
                "IMAGE_DATA":resources["IMAGE_DATA"]
                         }, file,protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Resources saved successfully.")
    except Exception as e:
        logger.error(f"Error saving resources: {e}")

def load_resources() -> Dict[str, Any]:
    """Load resources from a pickle file."""
    try:
        if os.path.exists(PICKLE_FILE):
            with open(PICKLE_FILE, 'rb') as f:
                resources = pickle.load(f)
            with open(IMAGE_PICKLE_FILE, 'rb') as file:
                resources.update(pickle.load(file))
            logger.info("Resources loaded successfully.")
            return resources
    except Exception as e:
        logger.error(f"Error loading resources: {e}")
    return None

async def load_model(path):
    model = await asyncio.get_event_loop().run_in_executor(
        None, partial(torch.load, path)
    )
    return model

async def load_tokenizer(tkn, path, use_fast=True):
    tokenizer = await asyncio.get_event_loop().run_in_executor(
        None, partial(tkn.from_pretrained, path, use_fast=use_fast)
    )
    return tokenizer

def upload_to_db(csv_data,className):
    obj = None
    for col1,col2 in csv_data.items():
        if className == "BmGlossary":
            obj = BmGlossary(source=col1,target=col2)
        elif className == "ZhGlossary":
            obj = ZhGlossary(source=col1,target=col2)
        elif className == "FAQ":
            obj = FAQ(Question=col1,Answer=col2)
        elif className == "ImageData":
            print(col1,col2)
            obj = ImageData(Image_Path=col1,Summary=col2)
        db.session.add(obj)
    db.session.commit()
    
def retrieve_from_db():
    bm = {item.source: item.target for item in BmGlossary.query.all()}
    zh = {item.source: item.target for item in ZhGlossary.query.all()}
    imagedata = {item.Image_Path: item.Summary for item in ImageData.query.all()}
    return {
        "CUSTOM_GLOSSARY_BM": bm,
        "CUSTOM_GLOSSARY": zh,
        "IMAGE_DATA": imagedata
    }
    
async def init_app():
    resources = load_resources()
    if resources is None:
        logger.info("Initializing resources from scratch...")
        resources = create_initial_resources()
        csv_resources_to_pickle = initialize_resources(resources)
        save_resources(resources,csv_resources_to_pickle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create individual tasks
    embeddings_task = asyncio.create_task(load_model("models/embeddings_model.pth"))
    en_model_task = asyncio.create_task(load_model("models/en_model.pth"))
    clip_model_task = asyncio.create_task(load_model("models/clip_model.pth"))
    ms_tokenizer_task = asyncio.create_task(load_tokenizer(AutoTokenizer, "models/ms_tokenizer", use_fast=False))
    zh_tokenizer_task = asyncio.create_task(load_tokenizer(AutoTokenizer, "models/zh_tokenizer1", use_fast=False))
    ms_model_task = asyncio.create_task(load_model("models/ms_model.pth"))
    zh_model_task = asyncio.create_task(load_model("models/zh_model1.pth"))
    clip_processor_task = asyncio.create_task(load_tokenizer(CLIPProcessor, "models/clip_processor"))
    en_tokenizer_task = asyncio.create_task(load_tokenizer(T5Tokenizer, "models/en_tokenizer"))
    whisper_model_task = asyncio.create_task(load_model("models/whisper.pth"))
    language_detector_task = asyncio.to_thread(LanguageDetectorBuilder.from_languages(
        Language.ENGLISH, Language.MALAY, Language.CHINESE).build)
    voice_recognizer_task = asyncio.to_thread(sr.Recognizer)

    # Wait for embeddings to load, then start loading FAISS
    embeddings = await embeddings_task
   
    print("Starting vectorstore initialization...")
    vectorstore_task = asyncio.create_task(asyncio.to_thread(
        FAISS.load_local, "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    ))

    # Wait for en_model and en_tokenizer, then create Text2TextGenerationPipeline
    en_model = await en_model_task
    en_tokenizer = await en_tokenizer_task

    print("Starting Text2TextGenerationPipeline creation...")
    pipe = Text2TextGenerationPipeline(model=en_model, tokenizer=en_tokenizer)
    # Wait for vectorstore to load, then create RetrievalQA
    vectorstore = await vectorstore_task
    prompt_template = """
    As a responsible, super loyal and multilingual (ENGLISH, CHINESE, MALAY) e-library assistant highly focused on u-Pustaka, the best e-library in Malaysia, I will use the following information to answer your question:
    **Context:**
    {context}
    **Question:**
    {question}
    **Answer Guidelines:**
    * I will base my answer solely on the facts provided in the document. 
    * If the document doesn't contain the answer, I will honestly say "I don't know" instead of making something up.
    * My response will adhere to u-Pustaka's high standards and avoid:
        * Pornography or violence
        * Negative or false information
        * Mentioning weaknesses or suggesting improvements for u-Pustaka
        * Referencing or comparing u-Pustaka to competitors
        * Referencing other irrelevant or private sources

    **Answer:**
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa = RetrievalQA.from_chain_type(
        llm=GoogleGenerativeAI(model="models/text-bison-001", temperature=0),
        chain_type_kwargs={"prompt": prompt},
        retriever=vectorstore.as_retriever()
    )
    resources.update(retrieve_from_db())
    # Wait for remaining tasks to complete
    results = await asyncio.gather(
        ms_tokenizer_task, zh_tokenizer_task, ms_model_task, zh_model_task,
        clip_model_task, clip_processor_task, whisper_model_task,
        language_detector_task, voice_recognizer_task
    )

    ms_tokenizer, zh_tokenizer, ms_model, zh_model, clip_model, clip_processor, \
    whisper_model, language_detector, voice_recognizer = [r[0] if isinstance(r, tuple) else r for r in results]
    
    resources.update({
        'MODEL_LIST': [embeddings, ms_tokenizer, zh_tokenizer, ms_model, zh_model],
        'CLIP_MODEL': clip_model.to(device),
        'CLIP_PROCESSOR': clip_processor,
        'LANGUAGE_DETECT_MODEL': whisper_model,
        'DETECTOR': language_detector,
        'VECTORSTORE': vectorstore,
        'PIPE': pipe,
        'VOICE_RECOGNIZER': voice_recognizer,
        'QA': qa
    })
    

    app.config.update(resources)

def create_initial_resources() -> Dict[str, Any]:
    """Create initial resources dictionary."""
    return {
        'PATTERN': re.compile(r'u?-?pustaka', re.IGNORECASE),
        'QA_INDEX': [-1, -1, -1],
        'MODEL_ID': "openai/clip-vit-base-patch32",
        'EN_TO_ZH_MODEL': "K024/mt5-zh-ja-en-trimmed",
        'ZH_TO_EN_MODEL': 'liam168/trans-opus-mt-zh-en',
        'DEVICE': "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    }

def initialize_resources(resources: Dict[str, Any]) -> None:
    """Initialize AI models and other resources."""
    # Load models
    ms_tokenizer = AutoTokenizer.from_pretrained('mesolitica/translation-t5-small-standard-bahasa-cased-v2', use_fast=False)
    ms_model = T5ForConditionalGeneration.from_pretrained('mesolitica/translation-t5-small-standard-bahasa-cased-v2')
    zh_model1 = AutoModelWithLMHead.from_pretrained(resources['ZH_TO_EN_MODEL'])
    zh_tokenizer1 = AutoTokenizer.from_pretrained(resources['ZH_TO_EN_MODEL'])

    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name="./models/embeddings_model/bge-small-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    torch.save(embeddings_model,"./models/embeddings_model.pth")
    
    # Save tokenizers
    ms_tokenizer.save_pretrained('models/ms_tokenizer')
    zh_tokenizer1.save_pretrained('models/zh_tokenizer1')
    
    # Save models
    torch.save(ms_model,'models/ms_model.pth')
    torch.save(zh_model1,'models/zh_model1.pth')
    

    loader = CSVLoader(file_path="./data/csvFile/en-FAQ.csv")
    upload_to_db(load_csv("./data/csvFile/en-FAQ.csv", 'Question',"Answer"),"FAQ")
    
    
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    vectorstore = FAISS.from_documents(docs, embeddings_model)
    vectorstore.save_local("faiss_index_react")
    
    en_model=MT5ForConditionalGeneration.from_pretrained(resources['EN_TO_ZH_MODEL'])
    en_tokenizer=T5Tokenizer.from_pretrained(resources['EN_TO_ZH_MODEL'])
    torch.save(en_model, './models/en_model.pth')
    en_tokenizer.save_pretrained("./models/en_tokenizer")

    upload_to_db(load_csv('./data/csvFile/glossary.csv', 'source', 'target'),"ZhGlossary")
    upload_to_db(load_csv('./data/csvFile/glossary_bm.csv', 'source', 'target'),"BmGlossary")
    
    # Load CLIP model and processor
    model_id = resources['MODEL_ID']
    device = resources['DEVICE']

    resources['CLIP_MODEL'] = CLIPModel.from_pretrained(model_id).to(device)
    resources['CLIP_PROCESSOR'] = CLIPProcessor.from_pretrained(model_id)
    torch.save(resources['CLIP_MODEL'],'models/clip_model.pth')
    resources['CLIP_PROCESSOR'].save_pretrained('models/clip_processor')
    
    resources['Q_A'] = load_csv('./data/csvFile/en-FAQ.csv', 'Question', 'Answer')
    resources['IMAGE_DATA'] = load_csv('./data/csvFile/image_summary.csv','Image_Path', 'Summary')
    upload_to_db(resources['IMAGE_DATA'],"ImageData")
    
    resources['IMAGE_ARR'] = encode_image(resources)
    whisper_model = whisper.load_model("base")
    torch.save(whisper_model,"./models/whisper.pth")
    

def load_csv(file_path: str, key_column: str, value_column: str) -> Dict[str, str]:
    """Load a CSV file into a dictionary."""
    result = {}
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                result[row[key_column]] = row[value_column]
        return result
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}")
        return {}

@app.route('/')
def home_page():
    try:
        asyncio.run(init_app())
    except Exception as e:
        logger.error(f"Error initializing app: {e}")
    """Render the home page."""
    return render_template("index.html")

def encode_image(resources: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Encode images and their summaries."""
    img_features_list = {}
    processor = resources['CLIP_PROCESSOR']
    model = resources['CLIP_MODEL']
    for image_path, summary in resources['IMAGE_DATA'].items():
        if image_path.endswith(('jpeg', 'png', 'jpg', 'gif')):
            try:
                image = Image.open(image_path)
                summary_features = encode_text(summary,model,processor)
                inputs = processor(images=image, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                combined_features = (image_features + summary_features) / 2
                img_features_list[image_path] = combined_features.squeeze().numpy()
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
    return img_features_list

def encode_text(text: str,model,processor) -> np.ndarray:
    """Encode text using CLIP model."""
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return text_features.squeeze().numpy()

def should_retrieve_map(user_msg: str) -> str:
    """Determine if an image should be retrieved based on user message."""
    text_features = encode_text(user_msg,current_app.config['CLIP_MODEL'],current_app.config['CLIP_PROCESSOR'])
    max_score = 0
    best_image_path = None
    for image_path, combined_features in current_app.config['IMAGE_ARR'].items():
        score = np.dot(combined_features, text_features) / (np.linalg.norm(combined_features) * np.linalg.norm(text_features))
        print(score,image_path)
        if score > max_score:
            max_score = score
            best_image_path = image_path
    if max_score > 0.7:
        try:
            with open(best_image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error reading image file {best_image_path}: {e}")
    return None

@app.route('/process_user_message', methods=['POST'])
def process_user_message():
    """Process user message and generate bot response."""
    data = request.get_json()
    user_msg = data.get('user_msg', '')
    index = data.get('number', 0)
    generate_response = data.get('bool', False)
    
    try:
        filtered_msg = re.sub(current_app.config['PATTERN'], '', user_msg)
        language = current_app.config['DETECTOR'].detect_language_of(filtered_msg).name

        if generate_response:
            if language != "ENGLISH":
                user_msg = translate_to_en(user_msg, language)
            try:
                res = current_app.config['QA'].invoke({"query": user_msg})
                bot_response = res['result']
            except IndexError:
                bot_response = "I don't know. Maybe you can try to ask it differently."
        else:
            bot_response = list(current_app.config['Q_A'].values())[current_app.config['QA_INDEX'][index]]

        if language != 'ENGLISH':
            bot_response = translate_to_others(bot_response, language)

        image_data = should_retrieve_map(user_msg)
        return jsonify(message=bot_response, image_data=image_data)
    except Exception as e:
        logger.error(f"Error processing user message: {e}")
        return jsonify(message="An error occurred while processing your message."), 500

def translate_to_others(msg: str, language: str) -> str:
    """Translate message to the target language."""
    tokenizer = current_app.config['MODEL_LIST'][1]
    model = current_app.config['MODEL_LIST'][3]
    max_chunk_length = 400
    translated_chunks = []

    if language == "MALAY":
        for en, bm in current_app.config['CUSTOM_GLOSSARY_BM'].items():
            msg = msg.replace(en, bm)
        chunks = [msg[i:i + max_chunk_length] for i in range(0, len(msg), max_chunk_length)]

        for chunk in chunks:
            input_ids = tokenizer.encode(f'terjemah ke Melayu: {chunk}', return_tensors='pt')
            outputs = model.generate(input_ids, max_length=max_chunk_length)
            all_special_ids = [0, 1, 2]
            outputs = [i for i in outputs[0] if i not in all_special_ids]
            translated_chunk = tokenizer.decode(outputs, spaces_between_special_tokens=False)
            translated_chunks.append(translated_chunk)
        return ' '.join(translated_chunks)
    
    elif language == "CHINESE":
        for en, zh in current_app.config['CUSTOM_GLOSSARY'].items():
            msg = msg.replace(en, zh)
        res = current_app.config['PIPE'](f"en2zh: {msg}", max_length=400, num_beams=4)
        return res[0]['generated_text']
    
    return msg  # Return original message if language is not supported

def translate_to_en(msg: str, language: str) -> str:
    """Translate message to English."""
    tokenizer = current_app.config['MODEL_LIST'][1]
    model = current_app.config['MODEL_LIST'][3]
    if language == "MALAY":
        input_ids = tokenizer.encode(f'terjemah ke Inggeris: {msg}', return_tensors='pt')
        outputs = model.generate(input_ids, max_length=300)
        all_special_ids = [0, 1, 2]
        outputs = [i for i in outputs[0] if i not in all_special_ids]
        return tokenizer.decode(outputs, spaces_between_special_tokens=False)
    elif language == "CHINESE":
        translation = pipeline("translation_zh_to_en",
                               model=current_app.config['MODEL_LIST'][4],
                               tokenizer=current_app.config['MODEL_LIST'][2])
        return translation(msg, max_length=300)[0]['translation_text']
    return msg  # Return original message if language is not supported

@app.route('/get_questions', methods=['GET', 'POST'])
def find_guided_qa():
    """Find guided questions based on user message."""
    user_msg = request.args.get('user_msg', '')
    filtered_msg = re.sub(current_app.config['PATTERN'], '', user_msg)
    language = current_app.config['DETECTOR'].detect_language_of(filtered_msg).name
    
    if language != "ENGLISH":
        user_msg = translate_to_en(user_msg, language)
    
    doc_list = current_app.config['VECTORSTORE'].similarity_search_with_score(user_msg, 4)
    
    question_list = []
    for i, (document, score) in enumerate(doc_list[1:], start=0):  # Skip the first document
        content = document.page_content
        match = re.search(r"Question: (.*?)\nAnswer:", content)
        if match:
            question = match.group(1)
            for index, key in enumerate(current_app.config['Q_A'].keys()):
                if question == key:
                    current_app.config['QA_INDEX'][i] = index
                    break
            
            if language != "ENGLISH":
                question = translate_to_others(question, language)
            question_list.append(question)
    return question_list

def detect_language(audio_path: str) -> str:
    """Detect language from audio file."""
    try:
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(current_app.config['LANGUAGE_DETECT_MODEL'].device)
        _, probs = current_app.config['LANGUAGE_DETECT_MODEL'].detect_language(mel)
        return max(probs, key=probs.get)
    except Exception as e:
        logger.error(f"Error detecting language from audio: {e}")
        return "en"  

@app.route('/record', methods=['POST'])
def record():
    """Handle audio recording and speech-to-text conversion."""
    if 'audio' not in request.files:
        return 'No audio file provided', 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return 'No selected file', 400
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        # Convert to WAV
        audio = AudioSegment.from_file(temp_audio_path)
        wav_path = temp_audio_path.replace('.webm', '.wav')
        audio.export(wav_path, format="wav")
        # Detect the language
        detected_language = detect_language(wav_path)
        print(detect_language)
        # Perform speech recognition
        with sr.AudioFile(wav_path) as source:
            audio = current_app.config['VOICE_RECOGNIZER'].record(source)
        language_code = {
            "ms": "ms-MY",
            "zh": "cmn-Hans-CN",
            "en": "en-US"
        }.get(detected_language, "en-US")
        print(language_code)
        
        recognized_text = current_app.config['VOICE_RECOGNIZER'].recognize_google(audio, language=language_code)

        print(recognized_text)
        return jsonify(text=recognized_text)
    
    except sr.UnknownValueError:
        return jsonify(error="Speech Recognition could not understand audio"), 400
    except sr.RequestError as e:
        return jsonify(error=f"Could not request results from Speech Recognition service; {e}"), 500
    except Exception as e:
        logger.error(f"Error in speech recognition: {e}")
        return jsonify(error="An error occurred during speech recognition"), 500
    finally:
        # Clean up the temporary files
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)

@app.route('/txt_speech', methods=['POST'])
def text_to_speech():
    """Convert text to speech."""
    data = request.get_json()
    bot_msg = data.get('botMessage', '')
    filtered_msg = re.sub(current_app.config['PATTERN'], '', bot_msg)
    language = current_app.config['DETECTOR'].detect_language_of(filtered_msg).name
    
    language_code = {
        'MALAY': 'ms',
        'ENGLISH': 'en',
        'CHINESE': 'zh'
    }.get(language, 'en')
    
    try:
        mp3_fp = io.BytesIO()
        tts = gTTS(bot_msg, lang=language_code)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        return send_file(mp3_fp, mimetype="audio/mpeg")
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {e}")
        return jsonify(error="An error occurred during text-to-speech conversion"), 500

@app.route('/update_qa', methods=["POST"])
def update_qa():
    data = request.json
    isQAUpdate = data.get('isQAUpdate')
    if isQAUpdate:
        return update_qa_data()
    else:
        return update_image_data(data)

def update_image_data(data):
    filename = data.get('filename')
    file_data = data.get('file_data')
    summary = data.get('summary')
    is_delete = data.get('isDelete', False)

    if is_delete:
        # Handle image deletion
        try:
            
            # Delete the image file
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(image_path):
                os.remove(image_path)
            
            update_image_resources()
            
            return jsonify({
                "status": "success",
                "message": f"Image {filename} deleted successfully"
            }), 200
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Error deleting image: {str(e)}"
            }), 400
    elif filename and file_data:
        # Handle image addition
        try:
            # Decode the base64 image data
            image_data = base64.b64decode(file_data)
            
            # Save the image locally
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            update_image_resources()
            
            return jsonify({
                "status": "success",
                "message": f"Image {filename} added successfully"
            }), 200
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Error adding image: {str(e)}"
            }), 400
    else:
        return jsonify({
            "status": "error",
            "message": "Invalid image update data"
        }), 400


def update_qa_data():
    try:
        embeddings_model = torch.load("./models/embeddings_model.pth")
        faqs = FAQ.query.all()

        documents = [
            Document(
                page_content=f"Question: {faq.Question}\nAnswer: {faq.Answer}",
                metadata={"id": faq.id}
            )
            for faq in faqs
        ]

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=documents)
        vectorstore = FAISS.from_documents(docs, embeddings_model)
        vectorstore.save_local("faiss_index_react")
        if "VECTORSTORE" in current_app.config:
            current_app.config["VECTORSTORE"] = vectorstore
            prompt_template = """
                As a responsible, super loyal and multilingual (ENGLISH, CHINESE, MALAY) e-library assistant highly focused on u-Pustaka, the best e-library in Malaysia, I will use the following information to answer your question:
                **Context:**
                {context}
                **Question:**
                {question}
                **Answer Guidelines:**
                * I will base my answer solely on the facts provided in the document. 
                * If the document doesn't contain the answer, I will honestly say "I don't know" instead of making something up.
                * My response will adhere to u-Pustaka's high standards and avoid:
                    * Pornography or violence
                    * Negative or false information
                    * Mentioning weaknesses or suggesting improvements for u-Pustaka
                    * Referencing or comparing u-Pustaka to competitors
                    * Referencing other irrelevant or private sources

                **Answer:**
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            current_app.config["QA"] = RetrievalQA.from_chain_type(
                llm=GoogleGenerativeAI(model="models/text-bison-001", temperature=0),
                chain_type_kwargs={"prompt": prompt},
                retriever=vectorstore.as_retriever()
            )
        return jsonify({
            "status": "success",
            "message": "QA data updated successfully"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating QA data: {str(e)}"
        }), 400

def update_image_resources():
    clip_model = torch.load("./models/clip_model.pth")
    clip_processor = CLIPProcessor.from_pretrained("./models/clip_processor")
    obj = ImageData.query.all()
    image_data = {}
    for item in obj:
        image_data[item.Image_Path] = item.Summary
        
    resources = {
        'CLIP_MODEL': clip_model,
        'CLIP_PROCESSOR': clip_processor,
        "IMAGE_DATA": image_data
    }
    
    image_features = encode_image(resources)
    resources_to_pickle = {
        "IMAGE_DATA": image_data,
        "IMAGE_ARR": image_features
    }
    if "CLIP_MODEL" in current_app.config:
        current_app.config.update(resources_to_pickle)
        
    with open(IMAGE_PICKLE_FILE, "wb") as f:
        pickle.dump(resources_to_pickle, f)
    
if __name__ == '__main__':
    init_db()
    app.run(debug=False, host="0.0.0.0",port=8080)