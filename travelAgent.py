from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import bs4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

llm = ChatOpenAI(model="gpt-3.5-turbo")

def researchAgent(query, llm):
    tools = load_tools(["ddg-search", "wikipedia"], llm=llm)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, prompt=prompt)
    webContext = agent_executor.invoke({"input": query})
    return webContext['output']

def loadData():
    loader = WebBaseLoader(
        web_paths=["https://www.dicasdeviagem.com/inglaterra/"],
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("postcontentwrap", "pagetitleloading")))
    )
    docs=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriver = vectorstore.as_retriever()
    return retriver

def getRelevantDocs(query):
    retriver = loadData()
    relevant_documents = retriver.invoke(query)
    return relevant_documents

def supervisorAgent(query, llm, webContext, relevant_documents):
    prompt_template = """
    você é um gerente de um agência de viagens, sua resposta final, deverá ser um roteiro de viagem, completo e detalhado!
    Utilize o contexto de eventos e preços de passagens, o input do usuário e também os documentos relevantes para elaborar o roteiro.
    Contexto: {webContext}
    Documento relevante: {relevant_documents}
    Usuario: {query}
    Assistente:
    """

    prompt = PromptTemplate(
        input_variables=["query", "webContext", "relevant_documents"],
        template=prompt_template
    )

    sequence = RunnableSequence(prompt | llm)

    response = sequence.invoke({
        "query": query,
        "webContext": webContext,
        "relevant_documents": relevant_documents
    })

    return response

def getResponse(query, llm):
    webContext = researchAgent(query, llm)
    relevant_documents = getRelevantDocs(query)
    response = supervisorAgent(query, llm, webContext, relevant_documents)
    return response

while True:
    print("\n\n")
    print("=========================================================\n")
    print("Bem vindo ao agência de viagens, via terminal!\n")
    print("Nossa especialidade é a inglaterra!\n")

    query = input(">> ")
    print("Perfeito!! nossa IA assumira daqui! \n")

    if query == "exit":
        break

    if query :
       print("Aguardando a resposta da nossa IA...\n\n")
       print(getResponse(query, llm).content)