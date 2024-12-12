import asyncio
import aiohttp
import os
import sys

from pymongo import MongoClient
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.gladia import GladiaSTTService
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.processors.filters.stt_mute_filter import STTMuteConfig, STTMuteFilter, STTMuteStrategy
from google_llm import GoogleLLMService
from pipecat.services.openai import OpenAILLMContext
from pipecat.transports.network.websocket_server import WebsocketServerParams, WebsocketServerTransport

from loguru import logger

from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

async def main():
    async with aiohttp.ClientSession() as session:
        transport = WebsocketServerTransport(
        host='0.0.0.0',
        params=WebsocketServerParams(
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            audio_sample_rate=24000,
        )
)

        # stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="H6QPv2pQZDcGqLwDTIJQ",
            optimize_streaming=True
        )
        # tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-helios-en")
        # tts = CartesiaTTSService(api_key=os.getenv("CARTESIA_API_KEY"), voice_id="ac7ee4fa-25db-420d-bfff-f590d740aeb2")
        stt = GladiaSTTService(
            api_key=os.getenv("GLADIA_API_KEY"),
        )

        stt_mute_filter = STTMuteFilter(
        stt_service=stt,
        config=STTMuteConfig(strategy=STTMuteStrategy.ALWAYS)
)

        llm = GoogleLLMService(
            model="gemini-1.5-flash-latest",
            # model="gemini-exp-1114",
            api_key=os.getenv("GOOGLE_API_KEY"),
        )
        client = MongoClient('mongodb+srv://pranay:sih2024@college-information.jtrhn.mongodb.net/')
        db = client['SIH']
        general_info_collection = db['general_information']
        courses_collection = db['courses']
        scholarships_collection = db['scholarships']
        cutoff_collection = db['cutoffs']
        events_collection = db['internalEvents']
        placement_collection = db['placements']
        exam_collection = db['entranceExams']
        async def check_colleges(function_name, tool_call_id, arguments, llm, context, result_callback):
            '''Return a list of currently listed colleges in the database from Rajasthan. Use when user wants to know about the colleges available. WHENEVER USER ASK WHAT ALL COLLEGES YOU KNOW ABOUT GIVE THIS LIST... FETCH DATA FROM DATA BASE. PROVIDE THE LIST OF AVAILABLE COLLEGES IN THE DATABASE'''
            colleges = list(general_info_collection.find())
            await result_callback(f"The colleges in the db are {colleges}")


        async def check_courses(function_name, tool_call_id, arguments, llm, context, result_callback):
            '''Return currently available courses in a particular college'''
            courses = list(courses_collection.find({'collegeName': arguments["name"]}))
            if courses:
                return await result_callback(f"The courses in the db are {courses}")
            else:
                return await result_callback({
                    "error": f"Sorry we don't have any information about courses in {arguments["name"]}"
                })
            
        async def check_fees(function_name, tool_call_id, arguments, llm, context, result_callback):
            '''Use this tool to get the courses in the college and their respective fees. All information is from the database'''
            courses = list(courses_collection.find({'collegeName': arguments["name"]}))
            return await result_callback(f"The courses and their fees in the db are {courses}")


        async def check_cutoff(function_name, tool_call_id, arguments, llm, context, result_callback):
            '''Return the cutoffs of all courses in a particular selected college'''
            cutoffs = list(cutoff_collection.find({'collegeName': arguments["name"]}))
        
            if cutoffs:
                return await result_callback(f"The cutoffs for the college {arguments["name"]} in the db are {cutoffs}")
            else:
                return await result_callback({
                    "error": f"Sorry, we don't have cutoff information for {arguments["name"]}"
                })
            
        async def check_scholarships(function_name, tool_call_id, arguments, llm, context, result_callback):
            '''Return the scholarships available in a particular selected college'''
            scholarships = list(scholarships_collection.find({'collegeName': arguments["name"]}))
        
            if scholarships:
                return await result_callback(f"The scholarships available in this college {arguments["name"]} in the db are {scholarships}")
            else:
                return await result_callback({
                    "error": f"Sorry, we don't have scholarship information for {arguments["name"]}"
                })
        async def check_events(function_name, tool_call_id, arguments, llm, context, result_callback):
            '''Return the events taking place in the particular college. Can be used as a promotion of a college or when the user asks for events'''
            events = list(events_collection.find({'collegeName': arguments["name"]}))
        
            if events:
                return await result_callback(f"The events available in this college {arguments["name"]} in the db are {events}")
            else:
                return await result_callback({
                    "error": f"Sorry, we don't have event information for {arguments["name"]}"
                })
            
        async def check_placements(function_name, tool_call_id, arguments, llm, context, result_callback):
            '''Return the placement hisory of the particular college. Can be used as a promotion of a college or when the user asks for placement data'''
            placements = list(placement_collection.find({'collegeName': arguments["name"]}))
        
            if placements:
                return await result_callback(f"The placements available in this college {arguments["name"]} in the db are {placements}")
            else:
                return await result_callback({
                    "error": f"Sorry, we don't have placements information for {arguments["name"]}"
                })
            
        async def check_exams(function_name, tool_call_id, arguments, llm, context, result_callback):
            '''Return the enterance exams required to get admission in a college.'''
            exams = list(exam_collection.find({'collegeName': arguments["name"]}))
        
            if exams:
                return await result_callback(f"The enterance exams to take admission this college {arguments["name"]} in the db are {exams}")
            else:
                return await result_callback({
                    "error": f"Sorry, we don't have exam information for {arguments["name"]}"
                })
        llm.register_function("check_colleges", check_colleges)
        llm.register_function("check_courses", check_courses)
        llm.register_function("check_fees", check_fees)
        llm.register_function("check_cutoff", check_cutoff)
        llm.register_function("check_scholarships", check_scholarships)
        llm.register_function("check_placements", check_placements)
        llm.register_function("check_events", check_events)
        llm.register_function("check_exams", check_exams)
        tools = [
            {
                "function_declarations": [
                    {
                        "name": "check_colleges",
                        "description": "Get the information about colleges in the database",
                        "parameters": {
                    "type": "OBJECT",  # Add type
                    "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "a string with value as rajasthan",
                                },
                            },
                            "required": ["location"],
                        },
                },
                    {
                        "name": "check_courses",
                        "description": "Return the courses for the specified college during the conversation",
                        "parameters": {
                    "type": "OBJECT",  # Add type
                    "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The college that was mentioned and the college that we need to get the courses for",
                                },
                            },
                            "required": ["name"],
                        },
                    },
                    {
                        "name": "check_fees",
                        "description": "Return the fees for the specified college during the conversation",
                        "parameters": {
                    "type": "OBJECT",  # Add type
                    "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The college that was mentioned and the college that we need to get the fees for",
                                },
                                "type": {
                                    "type": "string",
                                    "description": "The type of fees either management or regular. If not mentioned default to regular",
                                },
                            },
                            "required": ["name", "type"],
                        },
                    },
                    {
                        "name": "check_cutoff",
                        "description": "Return the cutoffs for the specified college during the conversation. when giving the cutoff please make sure you first ask the user the course and year and then provide for only that course and year",
                        "parameters": {
                    "type": "OBJECT",  # Add type
                    "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The college that was mentioned and the college that we need to get the cutoffs for",
                                },
                            },
                            "required": ["name"],
                        },
                    },
                    {
                        "name": "check_scholarships",
                        "description": "Return the scholarships for the specified college during the conversation",
                        "parameters": {
                    "type": "OBJECT",  # Add type
                    "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The college that was mentioned and the college that we need to get the scholarships for",
                                },
                            },
                            "required": ["name"],
                        },
                    },
                    {
                        "name": "check_events",
                        "description": "Return the events taking place in the particular college. Can be used as a promotion of a college or when the user asks for events",
                        "parameters": {
                    "type": "OBJECT",  # Add type
                    "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The college that was mentioned and the college that we need to get the events for",
                                },
                            },
                            "required": ["name"],
                        },
                    },
                    {
                        "name": "check_placements",
                        "description": "Return the placement hisory of the particular college. Can be used as a promotion of a college or when the user asks for placement data",
                        "parameters": {
                    "type": "OBJECT",  # Add type
                    "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The college that was mentioned and the college that we need to get the placements for",
                                },
                            },
                            "required": ["name"],
                        },
                    },
                    {
                        "name": "check_exams",
                        "description": "Return the enterance exams required to get admission in a college.",
                        "parameters": {
                    "type": "OBJECT",  # Add type
                    "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The college that was mentioned and the college that we need to get the enterance exams for",
                                },
                            },
                            "required": ["name"],
                        },
                    },
                ]
            }
        ]

        system_prompt = '''You are an AI-powered Student Assistance Chatbot for the Department of Technical Education, Government of Rajasthan. Your primary role is to provide accurate and helpful information about engineering and polytechnic institutes in Rajasthan.
ACCESSS THE COLLEGES INFO THROUGH THE TOOLS AND USE THE COLLEGE NAME. FETCH DATA FROM THE TOOLS ONLY ONLY ONLY.ACCESSS THE COLLEGES INFO THROUGH THE TOOLS AND USE THE COLLEGE NAME TO FETCH THE DATA FROM THE TOOLS. FETCH DATA FROM TOOLS ONLY ONLY ONLY
IF THE USER ASKS ABOUT ALL THE ENGINEERING COLLEGES AVAILABLE FETCH THE TOOL, SEE THE CATEGORY OF THE COLLEGES AVAILABLE THROUGH THE TOOL, AND TELL THE USER ABOUT THEM. 
IF THE USER ASKS ABOUT ALL THE POLYTECHNIC COLLEGES AVAILABLE FETCH THE TOOL, SEE THE CATEGORY OF THE COLLEGES AVAILABLE USING THE TOOL, AND TELL THE USER ABOUT THE POLYTECHNIC COLLEGES. 
IF THE USERASKS ABOUT SOME MEDICAL OR ARTS OR ANY OTHER MISCLENEOUS COLLEGES, JUST SAY YOU DONT HAVE ANY INFORMATION. IF THE USER DOES NOT MENTION TYPE OF COLLEGE DEFAULT TO ENGINEERING
THE COLLEGES IN RAJASTHAN ARE PRESENT IN THE DATABASE AND YOU CAN GET THEM BY USING THE TOOLS
YOU HAVE TO SPEAK IN THE LANGUAGE THAT THE USER ASKS FOR
FOR PLACEMENTS PLEASE ASK THE YEAR AND BRANCH BEFORE GIVING THE PLACEMENT INFORMATION TO THE USER
WHEN TALKING ABOUT MONEY/FEES THEN WRITE IT IN WORDS INSTEAD OF NUMBERS MAKES IT EASIER FOR IT TO BE MADE INTO VOICE
THE USER IS SPEAKING OUT THE COLLEGE NAME SO IF THE NAME DOES NOT MATCH EXACTLY BUT PARTIALLY WITH A COLLEGE NAME THEN CONFIRM IT WITH THE USER BEFORE PROCEEDING
PLEASE USE AS FEW CHARCTERS AS POSSIBLE TO AVOID SAYING TOO MUCH. FOR COLLEGE NAMES USE ABBRIVATIONS AND WHEN USER USES ABRIVATIONS THEN MATCH THEM WITH COLLEGE NAMES AND USE THE FULL NAME WITHOUT ABBRIVATIONS
Start the conversation by introducing yourself and asking how you can help with college information today. Always try to provide accurate, helpful, and efficient assistance to reduce the workload on department staff and enhance the user experience.
YOUR TEXT WILL BE CONVERTED INTO SPEECH/VOICE SO SAY EVERYTHING IN ONE LINE WITHOUT ANY PUNCHUATION MARKS OR EMOJIS OR ANYTHING THAT COULD SOUND UNNATUAL IN SPEECH. ALSO DON'T MAKE THE MESSAGE TOO BIG.
DO NOT USE ANY KIND OF MARKDOWN AS WELL OR ANY SYMBOL WHICH WOULD SOUND UNNATURAL IN SPEECH. KEEP EVERYTHING SHORT AND SIMPLE
ALSO NOTE BY DATABASE WE MEAN THE INFORMATION AVAILABLE THROUGH YOUR TOOLS. SO USE UR TOOLS EVERYTIME YOU NEED INFORMATION FROM THE DATABASE
'''
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Introduce yourself to the user"},
        ]

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                stt_mute_filter,
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, participant):
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())