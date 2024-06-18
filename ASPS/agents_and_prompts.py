PROMPT_INTRO = '''<s>You are a classroom assistant. Your job is to determine which of the following categories best fits a user's query. Please only provide the category number in your response. The available categories are as follows:'''

WIKI_AGENT_DESCRIPTION = '''The user is asking for factual information that could be found in a textbook or encyclopedia, including general description or explanations of terms and concepts.
For example: "What is the capital of France?", "Tell us about World War II", "What are protozoa?", "When did Darwin visit the Galapagos Islands?", "Please explain photosynthesis", "How tall is Mount Kilimanjaro?", "Why do we sneeze?", "What was the Roman empire?", "What mammal has the longest tail?", "Teach me about the Krebs cycle", "Who won the Spanish-American war?", "How far away is the sun?, "What is the pancreas?"'''

BING_AGENT_DESCRIPTION = '''The user is asking about current or recent events; contemporary pop culture; the year 2024 ('this year', 'currently', 'today') or 2023 ('last year')'; current seasons or current schedules; recent or upcoming elections; recent office-takers and award-winners; the age of living people; or any other information that is likely to have updated in the past 3 years. 
For example:  "When is Memorial Day in 2024?", "What is the latest news?", "What happened in politics last week?", "How much does a Tesla Model 3 currently cost?", "Who is Madonna currently dating?", "When does the next Dune movie come out?, "What is Kendrick Lamar's latest album?"
Only use this category for queries about events in the past 3 years, not historical figures or events.'''

GEMINI_AGENT_DESCRIPTION = '''The user is engaging in general conversation, including but not limited to: personal chat; matters of taste or opinion; solving math problems or converting units; map directions and distances between cities; requests for creative assistance or writing; requests for images, videos or resources; requests for lesson plans, classroom advice, activities, and quizzes; requests for translation, jokes, or predictions.
For example: "What is 26 miles in kilometers?", "Show me pictures of dogs", "What is the square root of 9?", "How do you say 'welcome' in Spanish?", "Solve for x: x + 7x = 16", "Who is the best football player?", "How long does it take to fly from Los Angeles to Seattle?", "Write me a letter of recommendation", "Who will win the next world cup?", "Is 3/5 greater than 1/2?", "How many liters are in 5 gallons?", "Give me a lesson plan for 4th grade civics", "How are you?", "What should I do for my birthday?", "Write me a haiku"'''


AGENT_DESRCIPTIONS_DICT = {
    "wiki": WIKI_AGENT_DESCRIPTION,
    "bing": BING_AGENT_DESCRIPTION,
    "gemini": GEMINI_AGENT_DESCRIPTION,
}

AGENT_COMBO_CASES = {
    1: ["wiki", "bing", "gemini"],
    2: ["wiki", "gemini", "bing"],
    3: ["bing", "gemini", "wiki"],
    4: ["bing", "wiki", "gemini"],
    5: ["gemini", "bing", "wiki"],
    6: ["gemini", "wiki", "bing"],

    7: ["wiki", "gemini"],
    8: ["gemini", "wiki"],

    9: ["bing", "gemini"],
    10: ["gemini", "bing"],

    11: ["wiki", "bing"],
    12: ["bing", "wiki"],
}