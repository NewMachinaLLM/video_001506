import certifi
import urllib3
import csv
from langchain_core.tools import tool
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

class MarineForecast:
    def __init__(self):
        self.wave_height = 0
        self.wave_period = 0
        return
    def getHumanReadableStr(self):
        return "The waves are "+str(self.wave_height)+" feet with period of "+str(self.wave_period)+" seconds."
  
# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

@tool
def getMarineForcast(buoyId: str) -> str:
    """
    Calls the US National Weather Service to get the marine forecast offshore at the buoyId.

    Args:
        buoyId (str): The buoyId to reference when get marine forecast information.

    Returns:
        MarineForecast: The string object has important marine forecast wave height and period information.
    """  
    # Get marineweather buoy data
    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
    response = http.request('GET', 'http://www.ndbc.noaa.gov/data/realtime2/'+ buoyId +'.txt')
    lines = response.data.decode("utf-8").splitlines()
    reader = csv.reader(lines)
    next(reader, None)  # Skip the first header
    next(reader, None)  # Skip the second header

    marine_forecast = MarineForecast()
    for row in reader:
        rowDatum = row[0]
        rowList = rowDatum.split()
        if rowList and rowList[8] != 'MM' and rowList[9] != 'MM':  # Ensure row is valid
            marine_forecast.wave_height = round( float(rowList[8]) * 3.28084 )
            marine_forecast.wave_period = rowList[9]
            break
    
    return marine_forecast.getHumanReadableStr()

# Augment the LLM with tools
tools = [getMarineForcast]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# Nodes
def llm_call(state: MessagesState):
    """LLM called"""

    return {
        "messages": [
            llm_with_tools.invoke(
            [
                SystemMessage(
                    content="You are a helpful assistant tasked with finding marine weather conditions. " +
                    "Here are guidelines on responding to marine forecast information prompts. " +
                    "<AnsweringGuidelines>" +
                    "   <Location>For marine forecasts of San Clemente California, check buoyId 46086.</Location>" +
                    "   <Location>For marine forecasts of Point Conception California, check buoyId 46054.</Location>" +
                    "   <Location>For marine forecasts of Mission Bay California, check buoyId 46258.</Location>" +
                    "   <Location>For marine forecasts of Point Loma California, check buoyId 46232.</Location>" +
                    "   <Location>For marine forecasts of Santa Monica Bay California, check buoyId 46221.</Location>" +
                    "   <Location>For marine forecasts of Half Moon Bay California, check buoyId 46214.</Location>" +
                    "   <Location>For marine forecasts of Mavericks in California, check buoyId 46214.</Location>" +
                    "</AnsweringGuidelines>"
                )
            ]
            + state["messages"]
            )
        ]
    }

def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["environment", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "Action"
    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "Action": "environment",
        END: END,
    },
)
agent_builder.add_edge("environment", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Invoke
messages = [HumanMessage(content="What are the wave heights and period off of Mavericks in HalfMoon Bay ?")]
#messages = [HumanMessage(content="What are the wave heights and period off of Santa Monica ?")]
#messages = [HumanMessage(content="What are the wave heights and period off of San Clemente ?")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()