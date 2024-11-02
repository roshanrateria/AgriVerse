import os
import numpy as np
import gradio as gr
import google.generativeai as genai
import joblib
from PIL import Image
import keras
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Load the crop prediction model
model = joblib.load('model.joblib')
# Load the crop recognizer model
crop_recognizer = keras.models.load_model('vggmodelweight.h5')  

# Configure the Gemini AI
genai.configure(api_key=os.getenv('API_KEY'))
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": None,
    "response_mime_type": "text/plain",
}
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)
chat_session = gemini_model.start_chat(history=[])


def get_readings():
    N = 104 
    P = 18
    K = 30
    temperature = 23.60
    humidity = 60.3
    ph = 6.7
    rainfall = 140.91
    return N, P, K, temperature, humidity, ph, rainfall

# Function to predict the crop
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    return prediction[0]
def get_commodity_data(url):
    # Make the request to the provided URL
    response = requests.post(url)
    
    # Parse the response content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table with class 'tableagmark_new'
    table = soup.find('table', {'class': 'tableagmark_new'})
    
    # Initialize an empty list to store the extracted data
    data = []
    
    # Find all rows in the table (ignore the header)
    rows = table.find_all('tr')[1:]  # Skipping the header row
    
    for row in rows:
        # Extract the table data (td) from each row
        cols = row.find_all('td')
        # Extracting and cleaning up text data from each column
        row_data = {
            "Sl no.": cols[0].text.strip(),
            "District Name": cols[1].text.strip(),
            "Market Name": cols[2].text.strip(),
            "Commodity": cols[3].text.strip(),
            "Variety": cols[4].text.strip(),
            "Grade": cols[5].text.strip(),
            "Min Price (Rs./Quintal)": cols[6].text.strip(),
            "Max Price (Rs./Quintal)": cols[7].text.strip(),
            "Modal Price (Rs./Quintal)": cols[8].text.strip(),
            "Price Date": cols[9].text.strip()
        }
        data.append(row_data)
    
    return data

# Example URL from your input
url = "https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity={}&Tx_State=0&Tx_District=0&Tx_Market=0&DateFrom={}&DateTo={}&Fr_Date={}&To_Date={}&Tx_Trend=0&Tx_CommodityHead=Coffee&Tx_StateHead=--Select--&Tx_DistrictHead=--Select--&Tx_MarketHead=--Select--"
# Get the processed commodity data
jute_url="https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity={}&Tx_State=0&Tx_District=0&Tx_Market=0&DateFrom={}&DateTo={}&Fr_Date={}&To_Date={}&Tx_Trend=0&Tx_CommodityHead=Coffee&Tx_StateHead=--Select--&Tx_DistrictHead=--Select--&Tx_MarketHead=--Select--"

def format_data_for_llm(commodity_data):
    # Initialize an empty string to store the formatted output
    formatted_text = "Commodity Price Data:\n\n"
    
    # Iterate over the list of dictionaries and format each entry
    for entry in commodity_data:
        formatted_text += (
            f"District Name: {entry['District Name']}\n"
            f"Market Name: {entry['Market Name']}\n"
            f"Commodity: {entry['Commodity']}\n"
            f"Variety: {entry['Variety']}\n"
            f"Grade: {entry['Grade']}\n"
            f"Min Price (Rs./Quintal): {entry['Min Price (Rs./Quintal)']}\n"
            f"Max Price (Rs./Quintal): {entry['Max Price (Rs./Quintal)']}\n"
            f"Modal Price (Rs./Quintal): {entry['Modal Price (Rs./Quintal)']}\n"
            f"Price Date: {entry['Price Date']}\n"
            f"{'-' * 40}\n"
        )
        break
    return formatted_text


# Function to recognize crop from image
comm={
    "Absinthe": 451,
    "Ajwan": 137,
    "Alasande Gram": 281,
    "Almond(Badam)": 325,
    'jute':16,
    
    "Alsandikai": 166,
    "Amaranthus": 86,
    "Ambada Seed": 130,
    "Ambady/Mesta": 417,
    "Amla(Nelli Kai)": 355,
    "Amphophalus": 102,
    "Amranthas Red": 419,
    "Antawala": 209,
    "Anthorium": 379,
    "Apple": 17,
    "Apricot(Jardalu/Khumani)": 326,
    "Arecanut(Betelnut/Supari)": 140,
    "Arhar (Tur/Red Gram)(Whole)": 49,
    "Arhar Dal(Tur Dal)": 260,
    "Asalia": 444,
    "Asgand": 505,
    "Ashgourd": 83,
    "Ashoka": 506,
    "Ashwagandha": 443,
    "Asparagus": 434,
    "Astera": 232,
    "Atis": 507,
    "Avare Dal": 269,
    "Bael": 418,
    "Bajji chilli": 491,
    "Bajra(Pearl Millet/Cumbu)": 28,
    "Balekai": 274,
    "balsam": 482,
    "Bamboo": 204,
    "Banana": 19,
    "Banana - Green": 90,
    "Banana flower": 483,
    "Banana Leaf": 485,
    "Banana stem": 484,
    "Barley (Jau)": 29,
    "basil": 435,
    "Bay leaf (Tejpatta)": 321,
    "Beans": 94,
    "Beaten Rice": 262,
    "Beetroot": 157,
    "Behada": 508,
    "Bengal Gram Dal (Chana Dal)": 263,
    "Bengal Gram(Gram)(Whole)": 6,
    "Ber(Zizyphus/Borehannu)": 357,
    "Betal Leaves": 143,
    "Betelnuts": 41,
    "Bhindi(Ladies Finger)": 85,
    "Bhui Amlaya": 448,
    "Big Gram": 113,
    "Binoula": 51,
    "Bitter gourd": 81,
    "Black Gram (Urd Beans)(Whole)": 8,
    "Black Gram Dal (Urd Dal)": 264,
    "Black pepper": 38,
    "BOP": 380,
    "Borehannu": 189,
    "Bottle gourd": 82,
    "Brahmi": 449,
    "Bran": 290,
    "Bread Fruit": 497,
    "Brinjal": 35,
    "Brocoli": 487,
    "Broken Rice": 293,
    "Broomstick(Flower Broom)": 320,
    "Bull": 214,
    "Bullar": 284,
    "Bunch Beans": 224,
    "Butter": 272,
    "buttery": 416,
    "Cabbage": 154,
    "Calendula": 480,
    "Calf": 215,
    "Camel Hair": 354,
    "Cane": 205,
    "Capsicum": 164,
    "Cardamoms": 40,
    "Carnation": 375,
    "Carrot": 153,
    "Cashew Kernnel": 238,
    "Cashewnuts": 36,
    "Castor Oil": 270,
    "Castor Seed": 123,
    "Cauliflower": 34,
    "Chakotha": 188,
    "Chandrashoor": 438,
    "Chapparad Avare": 169,
    "Chennangi (Whole)": 241,
    "Chennangi Dal": 295,
    "Cherry": 328,
    "Chikoos(Sapota)": 71,
    "Chili Red": 26,
    "Chilly Capsicum": 88,
    "Chironji": 509,
    "Chow Chow": 167,
    "Chrysanthemum": 402,
    "Chrysanthemum(Loose)": 231,
    "Cinamon(Dalchini)": 316,
    "cineraria": 467,
    "Clarkia": 478,
    "Cloves": 105,
    "Cluster beans": 80,
    "Coca": 315,
    "Cock": 368,
    "Cocoa": 104,
    "Coconut": 138,
    "Coconut Oil": 266,
    "Coconut Seed": 112,
    "coffee": 45,
    "Colacasia": 318,
    "Copra": 129,
    "Coriander(Leaves)": 43,
    "Corriander seed": 108,
    "Cossandra": 472,
    "Cotton": 15,
    "Cotton Seed": 99,
    "Cow": 212,
    "Cowpea (Lobia/Karamani)": 92,
    "Cowpea(Veg)": 89,
    "Cucumbar(Kheera)": 159,
    "Cummin Seed(Jeera)": 42,
    "Curry Leaf": 486,
    "Custard Apple (Sharifa)": 352,
    "Daila(Chandni)": 382,
    "Dal (Avare)": 91,
    "Dalda": 273,
    "Delha": 410,
    "Dhaincha": 69,
    "dhawai flowers": 442,
    "dianthus": 476,
    "Double Beans": 492,
    "Dragon fruit": 495,
    "dried mango": 423,
    "Drumstick": 168,
    "Dry Chillies": 132,
    "Dry Fodder": 345,
    "Dry Grapes": 278,
    "Duck": 370,
    "Duster Beans": 163,
    "Egg": 367,
    "Egypian Clover(Barseem)": 361,
    "Elephant Yam (Suran)": 296,
    "Field Pea": 64,
    "Fig(Anjura/Anjeer)": 221,
    "Firewood": 206,
    "Fish": 366,
    "Flax seeds": 510,
    "Flower Broom": 365,
    "Foxtail Millet(Navane)": 121,
    "French Beans (Frasbean)": 298,
    "Galgal(Lemon)": 350,
    "Gamphrena": 471,
    "Garlic": 25,
    "Ghee": 249,
    "Giloy": 452,
    "Gingelly Oil": 276,
    "Ginger(Dry)": 27,
    "Ginger(Green)": 103,
    "Gladiolus Bulb": 364,
    "Gladiolus Cut Flower": 363,
    "Glardia": 462,
    "Goat": 219,
    "Goat Hair": 353,
    "golden rod": 475,
    "Gond": 511,
    "Goose berry (Nellikkai)": 494,
    "Gram Raw(Chholia)": 359,
    "Gramflour": 294,
    "Grapes": 22,
    "Green Avare (W)": 165,
    "Green Chilli": 87,
    "Green Fodder": 346,
    "Green Gram (Moong)(Whole)": 9,
    "Green Gram Dal (Moong Dal)": 265,
    "Green Peas": 50,
    "Ground Nut Oil": 267,
    "Ground Nut Seed": 268,
    "Groundnut": 10,
    "Groundnut (Split)": 314,
    "Groundnut pods (raw)": 312,
    "Guar": 75,
    "Guar Seed(Cluster Beans Seed)": 413,
    "Guava": 185,
    "Gudmar": 453,
    "Guggal": 454,
    "gulli": 461,
    "Gur(Jaggery)": 74,
    "Gurellu": 279,
    "gypsophila": 469,
    "Haralekai": 252,
    "Harrah": 512,
    "He Buffalo": 216,
    "Heliconia species": 474,
    "Hen": 369,
    "Hippe Seed": 125,
    "Honey": 236,
    "Honge seed": 124,
    "Hybrid Cumbu": 119,
    "hydrangea": 473,
    "Indian Beans (Seam)": 299,
    "Indian Colza(Sarson)": 344,
}

def recognize_crop(image):
    pdict = {0: "jute", 1: "maize", 2: "rice", 3: "sugarcane", 4: "wheat"}
    img = Image.open(image).resize((224, 224))
    img = np.asarray(img) / 255.0
    img = img.reshape(-1, 224, 224, 3)
    prediction = crop_recognizer.predict(img)
    pred = np.argmax(prediction[0])
    return pdict[pred], prediction[0, pred] * 100
def parse(history):
    p='\n'
    if history==[]:
        return "No Chat till now"
    for i in history:
        try:
            p+=i['role']+': '+i['content']+'\n'
        except:
            print(i)
    return p
# Function to handle chatbot interaction and update readings/predictions
def chatbot_response(messages, prompt, readings_output):
    query = prompt.get("text")
    if query:
        N, P, K, temperature, humidity, ph, rainfall = get_readings()
        predicted_crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        recognized_crop, confidence = recognize_crop("jute-field.jpg")
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        tstr = today.strftime("%d-%b-%Y")
        ystr = yesterday.strftime("%d-%b-%Y")
        commodity_data = get_commodity_data(url.format(comm[recognized_crop],ystr,tstr,ystr,tstr))
        formatted_text = format_data_for_llm(commodity_data)
        comm_crop=get_commodity_data(jute_url.format(comm[predicted_crop],ystr,tstr,ystr,tstr))
        formatted_jute = format_data_for_llm(comm_crop)
        readings_output = [
            f"- Nitrogen (N): {N}",
            f"Phosphorus (P): {P}",
            f"Potassium (K): {K}",
            f"Temperature (°C): {temperature}",
            f"Humidity (%): {humidity}",
            f"pH: {ph}",
            f"Rainfall (mm): {rainfall}",
            f"Predicted Crop: {predicted_crop}", 
            f"Recognized Crops in place: {recognized_crop} ({confidence:.2f}%)"
        ]

        prompt_text = f'''
        You are an Agribot, an expert in sustainable agriculture. Assist farmers by answering their queries and suggesting organic solutions to maximize crop yield and profitability. Respond in the same language as of the query.
Soil Details:
Nitrogen (N): {N}
Phosphorous (P): {P}
Potassium (K): {K}
Temperature: {temperature}°C
Humidity: {humidity}%
pH: {ph}
Rainfall: {rainfall} mm
AI-Recommended Crop (Max Yield & Profit): {predicted_crop}
Most Recent Data for Recommended Crop :{formatted_text}
Recognized Crop in Field : {recognized_crop} 
Most Recent Data for Recognized Crop :{formatted_jute}
Today's Date: {tstr}
User History: {parse(messages)}
Query: {query}
        '''
        response = chat_session.send_message(prompt_text)

        messages.append(gr.ChatMessage(role="user", content=query))
        messages.append(gr.ChatMessage(role="assistant", content=response.text))
    return messages, gr.MultimodalTextbox(value=None, interactive=True),'\n - '.join( readings_output)

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"),fill_height=True) as demo:
    gr.Markdown("""# 🌱 AgriVerse 🌾
    ### For Proof of Concept ONLY.The final product will take *live* readings and images from field.
    ### It already Fetches live Prices data(of predicted and recognized crops ONLY) from *https://agmarknet.gov.in/* """)
    with gr.Row():
        
        with gr.Column(scale=2):
            readings_output = gr.Markdown("""
            - Nitrogen (N): 
            - Phosphorus (P): 
            - Potassium (K): 
            - Temperature (°C): 
            - Humidity (%): 
            - pH: 
            - Rainfall (mm):
            - Predicted Crop:  
            - Recognized Crops in place: 
            """,label="Readings & Predictions")
            image = gr.Image("jute-field.jpg", label="Image(live from field)",height="50%")
        with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", avatar_images=("user.jpeg", "logo.jpeg"), bubble_full_width=True)
                query_input = gr.MultimodalTextbox(interactive=True, placeholder="Enter message...", show_label=False)
                
                query_input.submit(
                    chatbot_response, 
                    [chatbot, query_input, readings_output], 
                    [chatbot, query_input, readings_output]
                )

# Launch the interface
demo.launch()
