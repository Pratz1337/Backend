from flask import Blueprint, jsonify, request
from flask_pymongo import PyMongo
from pymongo.errors import ConnectionFailure
from bson import ObjectId
from chat_model import *
import random
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from bson.objectid import ObjectId
import random
import os
from werkzeug.utils import secure_filename


# Blueprint for sidebar-related routes
sidebar_bp = Blueprint('sidebar', __name__)

# Helper function to convert ObjectId to string
def convert_objectid(document):
    document["_id"] = str(document["_id"])
    return document

# Initialize the ChatGoogleGenerativeAI model
summary_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5,
    max_tokens=800,
    timeout=None,
    max_retries=2,
    google_api_key="AIzaSyDPMDPp221VN3OznFnYj74ga0gDCPVxbEA"
)

# MongoDB configuration
MONGODB_URI = os.environ.get(
    'MONGODB_URI',
    'mongodb+srv://pranay:sih2024@college-information.jtrhn.mongodb.net/'
)
mongo = None  # Flask-PyMongo instance
db = None  # MongoClient instance

def init_mongo(app):
    """
    Initialize the MongoDB connection using Flask-PyMongo and MongoClient.
    """
    global mongo, db
    app.config["MONGO_URI"] = MONGODB_URI
    mongo = PyMongo(app)
    db = mongo.cx['SIH']  # Using MongoClient's connection for direct access

    try:
        # Ensure database connection
        mongo.cx.admin.command('ping')  # Using PyMongo's MongoClient connection
        print("Connected to MongoDB")
    except ConnectionFailure:
        print("Failed to connect to MongoDB")

# Quiz Questions Route
@sidebar_bp.route('/quiz/questions', methods=['GET'])
def get_quiz_questions():
    """
    Fetch a random set of 5 quiz questions.
    """
    try:
        # Ensure the collection exists
        if 'quiz_questions' not in db.list_collection_names():
            return jsonify({"error": "Quiz questions collection does not exist"}), 404

        # Fetch all quiz questions
        questions = list(db.quiz_questions.find())
        if not questions:
            return jsonify({"error": "No quiz questions found"}), 404

        # Convert ObjectId to string and shuffle questions
        questions_list = [convert_objectid(question) for question in questions]
        random.shuffle(questions_list)
        return jsonify(questions_list[:5]), 200
    except Exception as e:
        print(f"Error accessing quiz questions: {e}")
        return jsonify({"error": str(e)}), 500


# Quiz Submission Route
@sidebar_bp.route('/quiz/submit', methods=['POST'])
def submit_quiz():
    try:
        data = request.json
        answers = data['answers']
        questions = data['questions']
        
        # Predefined list of courses
        ALLOWED_COURSES = [
            "Computer Science and Engineering (CSE)",
            "Mechanical Engineering",
            "Civil Engineering",
            "Electrical Engineering",
            "Electronics and Communication Engineering (ECE)",
            "Information Technology (IT)",
            "Chemical Engineering",
            "Aerospace Engineering",
            "Artificial Intelligence (AI) and Machine Learning (ML)",
            "Biotechnology"
        ]
        
        # Prepare data for Gemini LLM
        llm_input = "\n".join([
            f"Question {i+1}: {q['question']}\nAnswer: {answers.get(q['_id'], 'No answer')}"
            for i, q in enumerate(questions)
        ])
        
        # Initialize Gemini LLM
        summary_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.5,
            max_tokens=200,
            timeout=None,
            max_retries=2,
            google_api_key="AIzaSyDPMDPp221VN3OznFnYj74ga0gDCPVxbEA "
        )
        
        # Create a prompt template for concise recommendations
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are an expert career counselor. Based on the quiz answers, recommend the top 2 courses from this list: {ALLOWED_COURSES}. Provide a brief, one-line reason for each recommendation. Format your response as: 'Course1: Reason1\nCourse2: Reason2'"),
            ("human", "Analyse these answers:\n{input}")
        ])
        
        # Create output parser
        output_parser = StrOutputParser()
        
        # Create the chain
        chain = prompt | summary_llm | output_parser
        
        # Generate LLM response
        gemini_recommendations = chain.invoke({"input": llm_input})
        
        # Print Gemini recommendations to terminal
        print("\n--- GEMINI LLM RECOMMENDATIONS ---")
        print(gemini_recommendations)
        
        return jsonify({
            "gemini_recommendations": gemini_recommendations
        }), 200
        
    except Exception as e:
        print(f"Error processing quiz submission: {e}")
        return {"error": str(e)}, 500


def analyze_quiz_answers(answers):
    course_scores = {
        "Computer Science and Engineering (CSE)": 0,
        "Mechanical Engineering": 0,
        "Electrical Engineering": 0,
        "Civil Engineering": 0,
        "Artificial Intelligence (AI)": 0,
        "Data Science": 0,
        "Biomedical Engineering": 0,
        "Aerospace Engineering": 0,
        "Chemical Engineering": 0,
        "Electronics and Communication Engineering": 0,
        "Information Technology (IT)": 0
    }
    
    for question_id, answer in answers.items():
        question = mongo.db.quiz_questions.find_one({"_id": ObjectId(question_id)})
        if question:
            weights = question.get("courseWeights", {})
            for course, weight in weights.items():
                if course in course_scores:
                    course_scores[course] += weight
    
    sorted_courses = sorted(course_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    recommendations = [
        {
            "course": course,
            "score": score,
            "reason": f"Based on your answers, you show a strong aptitude for {course}."
        }
        for course, score in sorted_courses if score > 0
    ]
    
    return recommendations

# The analyze_quiz_answers function remains the same as in the previous code
available_courses = [
    {
        "name": "Computer Science and Engineering",
        "details": "A comprehensive program focusing on advanced computing technologies and software development.",
        "scope": "Broad range of computing domains including software engineering, cloud computing, cybersecurity",
        "career_opportunities": [
            "Software Developer", 
            "Data Scientist", 
            "Cloud Architect", 
            "Cybersecurity Specialist"
        ],
        "skills_required": [
            "Programming (Python, Java, C++)", 
            "Data Structures", 
            "Algorithms", 
            "Cloud Technologies"
        ],
        "industry_demand": "Extremely High",
        "average_salary_inr": "8-12 LPA",
        "top_companies": ["TCS", "Infosys", "Google India", "Microsoft India"]
    },
    {
        "name": "Mechanical Engineering",
        "details": "In-depth study of machine design, thermal systems, and manufacturing processes.",
        "scope": "Manufacturing, automotive, aerospace, robotics, and renewable energy sectors",
        "career_opportunities": [
            "Design Engineer", 
            "Manufacturing Manager", 
            "Robotics Engineer", 
            "Automotive Design Specialist"
        ],
        "skills_required": [
            "CAD/CAM", 
            "Thermodynamics", 
            "Machine Design", 
            "Materials Engineering"
        ],
        "industry_demand": "High",
        "average_salary_inr": "5-9 LPA",
        "top_companies": ["Tata Motors", "Mahindra", "ISRO", "Automotive Research Centers"]
    },
    {
        "name": "Data Science",
        "details": "Advanced analytics, machine learning, and big data technologies.",
        "scope": "Analytics, AI, machine learning, business intelligence",
        "career_opportunities": [
            "Data Analyst", 
            "Machine Learning Engineer", 
            "Business Intelligence Specialist", 
            "AI Research Scientist"
        ],
        "skills_required": [
            "Python", 
            "R", 
            "Machine Learning", 
            "Statistical Analysis", 
            "Data Visualization"
        ],
        "industry_demand": "Very High",
        "average_salary_inr": "7-15 LPA",
        "top_companies": ["Walmart India", "Amazon", "Flipkart", "Reliance"]
    },
    {
        "name": "Artificial Intelligence",
        "details": "Cutting-edge research in neural networks, natural language processing, and intelligent systems.",
        "scope": "AI research, machine learning, robotics, natural language processing",
        "career_opportunities": [
            "AI Research Scientist", 
            "NLP Engineer", 
            "Computer Vision Specialist", 
            "Robotics Engineer"
        ],
        "skills_required": [
            "Deep Learning", 
            "Neural Networks", 
            "Python", 
            "TensorFlow", 
            "Natural Language Processing"
        ],
        "industry_demand": "Extremely High",
        "average_salary_inr": "10-20 LPA",
        "top_companies": ["Google AI", "Microsoft Research", "IBM India", "Intel India"]
    },
    {
        "name": "Electrical Engineering",
        "details": "Advanced study of electrical systems, power engineering, and electronics.",
        "scope": "Power generation, telecommunications, electronics, renewable energy",
        "career_opportunities": [
            "Power Systems Engineer", 
            "Electronics Design Engineer", 
            "Telecommunications Specialist", 
            "Renewable Energy Consultant"
        ],
        "skills_required": [
            "Circuit Design", 
            "Power Systems", 
            "Electronics", 
            "Embedded Systems"
        ],
        "industry_demand": "High",
        "average_salary_inr": "6-10 LPA",
        "top_companies": ["Power Grid Corporation", "BHEL", "Siemens", "Schneider Electric"]
    },
    {
        "name": "Biotechnology",
        "details": "Innovative research at the intersection of biology and technology.",
        "scope": "Pharmaceuticals, medical research, genetic engineering, agriculture",
        "career_opportunities": [
            "Research Scientist", 
            "Bioprocess Engineer", 
            "Genetic Counselor", 
            "Pharmaceutical Researcher"
        ],
        "skills_required": [
            "Molecular Biology", 
            "Genetic Engineering", 
            "Lab Techniques", 
            "Bioinformatics"
        ],
        "industry_demand": "Growing",
        "average_salary_inr": "5-12 LPA",
        "top_companies": ["Biocon", "Dr. Reddy's", "Serum Institute", "CSIR Labs"]
    }
]

@sidebar_bp.route("/get-courses", methods=["GET"])
def get_courses():
    """
    Fetch the list of available courses for the dropdown.
    """
    try:
        return jsonify({"courses": [course["name"] for course in available_courses]}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to fetch courses: {str(e)}"}), 500

# Compare Courses Route
@sidebar_bp.route("/compare-courses", methods=["POST"])
def compare_courses():
    try:
        data = request.json
        course1_name = data.get("course1")
        course2_name = data.get("course2")
        extra_field = data.get("extra_field", None)

        if not course1_name or not course2_name:
            return jsonify({"error": "Both course1 and course2 are required"}), 400

        course1 = next((course for course in available_courses if course["name"] == course1_name), None)
        course2 = next((course for course in available_courses if course["name"] == course2_name), None)

        if not course1 or not course2:
            return jsonify({"error": "One or both selected courses are invalid"}), 400

        # Generate a comprehensive AI-powered comparison
        prompt = f"""
Provide a detailed, comparative analysis of {course1_name} and {course2_name} in the following key areas, formatted as short bullet points:

1. **Scope and Domain**: 
   - Briefly highlight the scope of each course and the domain it covers.
   
2. **Career Prospects in India**: 
   - Mention the key career opportunities available for each course in India.
   
3. **Required Skills and Competencies**: 
   - List the essential skills and competencies needed for each course.

4. **Industry Demand and Growth Potential**: 
   - Summarize the demand for each course in the industry and the growth potential.

5. **Typical Salary Ranges and Career Progression**: 
   - Provide typical salary ranges for each course and potential career progression.

Use a clear and engaging bullet-point format for easy reading, ensuring each point is short but informative.

If the extra_field is provided:
    - Add a brief bullet point comparison on the aspect of '{extra_field}' for both courses.
"""


        if extra_field:
            prompt += f"\nAdditionally, provide insights on '{extra_field}' comparing both courses."

        # AI-generated comparison (optional, can be replaced with a more sophisticated prompt)
        response = summary_llm.invoke(prompt)
        ai_response = response.content if hasattr(response, "content") else str(response)

        return jsonify({
            "comparison": ai_response,
            "course1_details": {
                "name": course1["name"],
                "career_opportunities": course1["career_opportunities"],
                "skills_required": course1["skills_required"],
                "industry_demand": course1["industry_demand"],
                "average_salary_inr": course1["average_salary_inr"],
                "top_companies": course1["top_companies"],
            },
            "course2_details": {
                "name": course2["name"],
                "career_opportunities": course2["career_opportunities"],
                "skills_required": course2["skills_required"],
                "industry_demand": course2["industry_demand"],
                "average_salary_inr": course2["average_salary_inr"],
                "top_companies": course2["top_companies"],
            },
        })

    except Exception as e:
        return jsonify({"error": f"Failed to compare courses: {str(e)}"}), 500
