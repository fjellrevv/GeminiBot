import telebot
from telebot import types
import PyPDF2
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Токен бота
bot = telebot.TeleBot("", parse_mode=None)

command = "start"

# Вставка ключа Gemini
os.environ["GOOGLE_API_KEY"] = ""

# Загрузка документа
text = ""
file = open("data/text.pdf", "rb")
pdf = PyPDF2.PdfReader(file)
for page in pdf.pages:
    text += page.extract_text()

# Разбиение документа на части
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300,
    length_function=len
)
texts = text_splitter.split_text(text)

# Создание векторного хранилища
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = FAISS.from_texts(texts, embedding=embeddings)
vector_store.as_retriever()
vector_store.save_local("faiss_local")

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.4, max_length=10000)
system_prompt = """This file contains data on 10 different people. Each person has a description of their age, 
    profession, salary, work experience, and additional information. All descriptions are numbered. Text written 
    under a person's name applies to that person until the next name.) Answer the question in as much detail as 
    possible, given the context provided. \n\nContext:\n {context}?\nQuestion: \n{question}\nAnswer:"""


@bot.message_handler(commands=['document'])
def get_document(message):
    global command
    command = 'start'
    print(text)
    bot.send_message(message.from_user.id, text)


@bot.message_handler(commands=['free_talk'])
def get_gemini_response_free_talk(message):
    global command
    command = 'free_talk'
    response = 'Общение с ботом в свободной форме началось. Отправляйте свой запрос.'
    print(response)
    bot.send_message(message.from_user.id, response)


@bot.message_handler(commands=['rag'])
def get_gemini_response_rag(message):
    global command
    command = 'rag'
    response = ('Отправляйте свой запрос, связанный с приложенным для тестирования документом.\nЧтобы увидеть его '
                'содержимое, выполните команду /document.')
    print(response)
    bot.send_message(message.from_user.id, response)


@bot.message_handler(commands=['start'])
def get_start(message):
    global command
    command = 'start'
    text = ('Задавай боту вопросы, касающихся документов, приложенных для тестирования, или пиши вопросы в свободной '
            'форме. Используй для этого следующие команды:\n/document - просмотр содержимого документа, вложенного в '
            'файл проекта.\n/free_talk - общение с ботом в свободной форме.\n/rag - поиск ответов по тестовому '
            'документу, вложенному в файл проекта.\n/start - стартовое сообщение.')
    print(text)
    bot.send_message(message.from_user.id, text)


@bot.message_handler(content_types=['text'])
def get_gemini_response(message):
    bot.send_chat_action(message.chat.id, 'typing')
    if command == 'free_talk':
        response = llm.invoke(message.text)
        print(response)
        bot.send_message(message.chat.id, response.content)

    if command == 'rag':
        # Обработка сообщения пользователя
        db = FAISS.load_local("faiss_local", embeddings, allow_dangerous_deserialization=True)
        info = db.similarity_search(message.text)

        # Инструкция и цепочки
        prompt = PromptTemplate(template=system_prompt, input_variables=["context", "question"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

        response = chain(
            {"input_documents": info, "question": message.text}
            , return_only_outputs=True)

        print(response)
        bot.send_message(message.from_user.id, response['output_text'])

    if command == 'start':
        get_start(message)


bot.polling(none_stop=1, interval=0)
