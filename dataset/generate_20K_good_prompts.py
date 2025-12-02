import random
import pandas as pd

# ==========================================
# 1. EXPANDED VARIABLES (Scale up for 20k)
# ==========================================
variables = {
    # --- Tech / Professional ---
    "ui_elements": [
        ("login screen", "screenshot of a minimal login form with email and password fields"),
        ("dashboard widget", "screenshot of a dark-mode analytics widget showing user retention stats"),
        ("checkout page", "screenshot of an e-commerce checkout flow with credit card inputs"),
        ("navbar", "screenshot of a responsive top navigation bar with dropdown menus"),
        ("settings modal", "screenshot of a modal window with toggle switches for privacy settings"),
        ("profile card", "screenshot of a user profile card with avatar and bio"),
        ("kanban board", "screenshot of a drag-and-drop kanban board with tasks"),
        ("music player", "screenshot of a music player UI with album art and controls"),
        ("weather widget", "screenshot of a weather widget showing temperature and forecast"),
        ("calculator app", "screenshot of a scientific calculator interface")
    ],
    "documents": [
        ("grocery receipt", "photo of a crumpled thermal receipt from a supermarket"),
        ("medical invoice", "scan of a hospital bill with line items and insurance codes"),
        ("lease agreement", "photo of a signed residential lease contract page"),
        ("handwritten note", "photo of a sticky note with a to-do list in cursive ink"),
        ("passport", "scan of a passport data page (synthetic/redacted)"),
        ("utility bill", "photo of an electricity bill with usage graphs"),
        ("resume", "image of a professional resume/CV"),
        ("restaurant menu", "photo of a laminated dinner menu"),
        ("event ticket", "photo of a concert ticket with QR code"),
        ("shipping label", "close-up of a FedEx shipping label with barcode")
    ],
    "charts": [
        ("bar chart", "image of a vertical bar chart comparing quarterly revenue"),
        ("heatmap", "image of a color-coded correlation matrix heatmap"),
        ("scatter plot", "image of a scatter plot showing distribution of customer ages"),
        ("line graph", "image of a line graph showing stock price trends"),
        ("pie chart", "image of a pie chart showing market share"),
        ("candlestick chart", "image of a forex trading chart with indicators"),
        ("gantt chart", "image of a project timeline gantt chart"),
        ("radar chart", "image of a spider/radar chart comparing skill sets"),
        ("funnel chart", "image of a sales funnel conversion chart"),
        ("histogram", "image of a histogram showing frequency distribution")
    ],

    # --- Culinary / Food ---
    "food_items": [
        ("sushi platter", "photo of a vibrant sushi plate with wasabi and ginger"),
        ("homemade lasagna", "close-up photo of a cheesy lasagna fresh out of the oven"),
        ("avocado toast", "top-down photo of avocado toast with poached eggs"),
        ("chocolate cake", "photo of a slice of rich chocolate cake on a white plate"),
        ("vegetable stir-fry", "photo of a colorful wok filled with stir-fried vegetables"),
        ("cocktail", "photo of a fancy cocktail with a garnish on a bar counter"),
        ("pizza margherita", "photo of a classic pizza margherita with basil"),
        ("ramen bowl", "photo of a steaming bowl of ramen with egg and pork"),
        ("fruit salad", "photo of a fresh fruit salad with melon and berries"),
        ("grilled salmon", "photo of a grilled salmon fillet with asparagus"),
        ("tacos", "photo of three street tacos with salsa and lime"),
        ("burger", "photo of a gourmet burger with fries"),
        ("croissant", "photo of a flaky croissant on a bakery shelf"),
        ("curry", "photo of a bowl of chicken curry with rice"),
        ("smoothie bowl", "photo of an acai bowl with granola and banana")
    ],

    # --- Travel / Landmarks ---
    "landmarks": [
        ("Eiffel Tower", "photo of the Eiffel Tower in Paris against a blue sky"),
        ("Great Wall of China", "photo of the Great Wall winding through mountains"),
        ("Taj Mahal", "photo of the Taj Mahal reflecting in the pool"),
        ("Grand Canyon", "wide-angle photo of the Grand Canyon at sunset"),
        ("Kyoto Shrine", "photo of a traditional Japanese wooden shrine with cherry blossoms"),
        ("Statue of Liberty", "photo of the Statue of Liberty in New York Harbor"),
        ("Colosseum", "photo of the Colosseum in Rome"),
        ("Pyramids of Giza", "photo of the Pyramids of Giza in the desert"),
        ("Machu Picchu", "photo of Machu Picchu ruins in the Andes"),
        ("Sydney Opera House", "photo of the Sydney Opera House at night"),
        ("Santorini", "photo of white houses with blue domes in Santorini"),
        ("Mount Fuji", "photo of Mount Fuji with snow cap"),
        ("Golden Gate Bridge", "photo of the Golden Gate Bridge in fog"),
        ("Big Ben", "photo of Big Ben and Parliament in London"),
        ("Petra", "photo of the Treasury at Petra carved into rock")
    ],

    # --- Fashion ---
    "fashion_items": [
        ("summer floral dress", "photo of a mannequin wearing a floral summer dress"),
        ("leather jacket", "photo of a black leather biker jacket"),
        ("sneakers", "side-profile photo of limited edition colorful sneakers"),
        ("vintage handbag", "photo of a brown leather vintage handbag"),
        ("formal suit", "photo of a charcoal grey men's suit"),
        ("denim jeans", "photo of a pair of distressed blue jeans"),
        ("sunglasses", "photo of aviator sunglasses on a table"),
        ("wristwatch", "close-up photo of a luxury mechanical wristwatch"),
        ("running shoes", "photo of bright neon running shoes"),
        ("winter coat", "photo of a beige wool trench coat"),
        ("yoga pants", "photo of black leggings/yoga pants"),
        ("fedora hat", "photo of a stylish felt fedora hat"),
        ("scarf", "photo of a knitted wool scarf"),
        ("high heels", "photo of red stiletto heels"),
        ("backpack", "photo of a rugged hiking backpack")
    ],

    # --- Nature / Biology ---
    "nature_subjects": [
        ("wild mushroom", "close-up macro photo of a red mushroom in the forest"),
        ("garden flower", "photo of a blooming yellow rose in a garden"),
        ("strange insect", "macro photo of a colorful beetle on a leaf"),
        ("bird", "photo of a blue jay perched on a tree branch"),
        ("constellation", "night sky photo showing the Orion constellation"),
        ("oak tree", "photo of a large oak tree in a park"),
        ("butterfly", "photo of a monarch butterfly on a flower"),
        ("spider web", "photo of a spider web with dew drops"),
        ("seashell", "photo of a conch shell on the beach"),
        ("fern", "photo of a green fern plant"),
        ("frog", "photo of a small green frog on a lily pad"),
        ("mountain goat", "photo of a mountain goat on a cliff"),
        ("cactus", "photo of a saguaro cactus in the desert"),
        ("coral reef", "underwater photo of colorful coral"),
        ("cloud formation", "photo of cumulonimbus clouds")
    ],

    # --- Education ---
    "educational_visuals": [
        ("water cycle diagram", "illustrated diagram showing evaporation, condensation, and precipitation"),
        ("geometry problem", "photo of a textbook page showing a triangle with missing angles"),
        ("cell structure", "labeled diagram of an animal cell"),
        ("historical map", "image of an old map of Europe from the 18th century"),
        ("sheet music", "scan of a piano sheet music page"),
        ("periodic table", "image of the periodic table of elements"),
        ("solar system model", "diagram of the planets orbiting the sun"),
        ("human skeleton", "diagram of the human skeletal system"),
        ("circuit diagram", "schematic diagram of a simple electrical circuit"),
        ("chemical reaction", "diagram showing a chemical equation"),
        ("food pyramid", "chart showing nutritional food groups"),
        ("tectonic plates", "map showing earth's tectonic plates"),
        ("photosynthesis", "diagram explaining the process of photosynthesis"),
        ("fraction chart", "visual chart showing fraction equivalents"),
        ("color wheel", "image of a color wheel for art theory")
    ],

    # --- Home / DIY ---
    "home_items": [
        ("mid-century chair", "photo of a wooden mid-century modern chair"),
        ("leaky faucet", "photo of a kitchen faucet dripping water"),
        ("houseplant", "photo of a large monstera plant in a ceramic pot"),
        ("messy bookshelf", "photo of a disorganized bookshelf filled with old books"),
        ("broken window", "photo of a cracked window pane"),
        ("stained carpet", "photo of a coffee stain on a beige carpet"),
        ("wall paint swatches", "photo of various blue paint swatches on a wall"),
        ("antique lamp", "photo of a brass tiffany-style lamp"),
        ("kitchen cabinet", "photo of a white shaker-style kitchen cabinet"),
        ("garden tool", "photo of a rusty garden trowel"),
        ("ceramic vase", "photo of a handmade blue ceramic vase"),
        ("wooden floor", "photo of hardwood flooring with a scratch"),
        ("ceiling fan", "photo of a modern ceiling fan"),
        ("smart thermostat", "photo of a digital smart thermostat on a wall"),
        ("power drill", "photo of a cordless power drill")
    ],

    # --- Independent Variables ---
    "tech_stacks": ["React + Tailwind", "Vue 3", "HTML/CSS", "Bootstrap", "Svelte", "Angular", "SwiftUI", "Kotlin/Jetpack Compose"],
    "output_formats": ["JSON", "CSV", "Markdown", "YAML", "XML", "SQL Insert Statements", "Plain Text"],
    "writing_styles": ["mystery", "sci-fi", "romantic", "melancholic", "humorous", "professional", "academic", "poetic", "journalistic", "persuasive"],
    "audiences": ["a 5-year-old", "a college student", "an expert", "a tourist", "a senior citizen", "a developer", "a chef", "a history buff"],
    "dietary_restrictions": ["vegan", "gluten-free", "keto", "low-carb", "dairy-free", "nut-free", "halal", "kosher"],
    "languages": ["Spanish", "French", "Japanese", "German", "Italian", "Portuguese", "Chinese", "Russian", "Arabic", "Hindi"]
}

# ==========================================
# 2. TEMPLATES (More templates for variety)
# ==========================================
templates = [
    # --- Coding ---
    {"category": "Coding", "text": "Act as a frontend developer. Convert this {ui_element} screenshot into production-ready {tech_stack}.", "context_source": "ui_elements", "context_key": "{ui_element}"},
    {"category": "Coding", "text": "Identify the UI components in this {ui_element} and list them as a JSON array for a design system.", "context_source": "ui_elements", "context_key": "{ui_element}"},
    
    # --- OCR ---
    {"category": "OCR/Extraction", "text": "Extract the text from this {document} and return it as {output_format}.", "context_source": "documents", "context_key": "{document}"},
    {"category": "OCR/Extraction", "text": "Find the date and total amount in this {document}. Return only these two values.", "context_source": "documents", "context_key": "{document}"},

    # --- Data Analysis ---
    {"category": "Data Analysis", "text": "Analyze this {chart}. Summarize the key trend for a business executive.", "context_source": "charts", "context_key": "{chart}"},
    {"category": "Data Analysis", "text": "Convert the data shown in this {chart} into a CSV format.", "context_source": "charts", "context_key": "{chart}"},

    # --- Culinary ---
    {"category": "Culinary/Food", "text": "Identify the dish in this image of {food_item}. Estimate the calories and suggest a {dietary_restriction} alternative.", "context_source": "food_items", "context_key": "{food_item}"},
    {"category": "Culinary/Food", "text": "I have these ingredients shown in the {food_item} image. What spice should I add to elevate the flavor?", "context_source": "food_items", "context_key": "{food_item}"},
    {"category": "Culinary/Food", "text": "Write a mouth-watering instagram caption for this photo of {food_item}.", "context_source": "food_items", "context_key": "{food_item}"},

    # --- Travel ---
    {"category": "Travel/Landmarks", "text": "Identify this landmark: {landmark}. Provide a 3-day itinerary for a trip here.", "context_source": "landmarks", "context_key": "{landmark}"},
    {"category": "Travel/Landmarks", "text": "Translate the signage visible in this photo of {landmark} into {languages} and explain the cultural context.", "context_source": "landmarks", "context_key": "{landmark}"},
    {"category": "Travel/Landmarks", "text": "What is the best time of year to visit {landmark} to take a photo like this?", "context_source": "landmarks", "context_key": "{landmark}"},

    # --- Fashion ---
    {"category": "Fashion/Style", "text": "Act as a stylist. Look at this {fashion_item}. What accessories would you pair with this for a formal event?", "context_source": "fashion_items", "context_key": "{fashion_item}"},
    {"category": "Fashion/Style", "text": "Identify the style and era of this {fashion_item}. Where could I buy something similar today?", "context_source": "fashion_items", "context_key": "{fashion_item}"},
    {"category": "Fashion/Style", "text": "Is this {fashion_item} appropriate for a business casual office environment?", "context_source": "fashion_items", "context_key": "{fashion_item}"},

    # --- Education ---
    {"category": "Education", "text": "Act as a tutor. Explain the concept shown in this {educational_visual} to {audiences}.", "context_source": "educational_visuals", "context_key": "{educational_visual}"},
    {"category": "Education", "text": "Solve the problem shown in this {educational_visual}. Show your work step-by-step.", "context_source": "educational_visuals", "context_key": "{educational_visual}"},
    {"category": "Education", "text": "Create a quiz question based on the information in this {educational_visual}.", "context_source": "educational_visuals", "context_key": "{educational_visual}"},

    # --- Nature ---
    {"category": "Nature/Biology", "text": "Identify this {nature_subject}. Is it poisonous or dangerous? Provide a scientific classification.", "context_source": "nature_subjects", "context_key": "{nature_subject}"},
    {"category": "Nature/Biology", "text": "Describe the habitat where you would typically find this {nature_subject}.", "context_source": "nature_subjects", "context_key": "{nature_subject}"},

    # --- Creative Writing ---
    {"category": "Creative Writing", "text": "Write a {writing_style} short story opening inspired by the atmosphere in this photo of {landmark}.", "context_source": "landmarks", "context_key": "{landmark}"},
    {"category": "Creative Writing", "text": "Describe this {nature_subject} using only metaphors and similes.", "context_source": "nature_subjects", "context_key": "{nature_subject}"},

    # --- Home/DIY ---
    {"category": "Home/DIY", "text": "Identify the style of this {home_item}. Suggest a color palette for a room that features this piece.", "context_source": "home_items", "context_key": "{home_item}"},
    {"category": "Home/DIY", "text": "Look at this {home_item}. What tools would I likely need to fix or restore it?", "context_source": "home_items", "context_key": "{home_item}"},
    {"category": "Home/DIY", "text": "How would you clean this {home_item} without damaging the material?", "context_source": "home_items", "context_key": "{home_item}"}
]

def generate_large_dataset(n=20000):
    data = []
    # Ensure exact N
    for i in range(n):
        template = random.choice(templates)
        prompt = template["text"]
        
        # 1. Context Variable Injection
        source_key = template["context_source"]
        placeholder = template["context_key"]
        
        selection_value, selection_context = random.choice(variables[source_key])
        prompt = prompt.replace(placeholder, selection_value)
        
        # 2. Independent Variable Injection
        # Iterate through all potential placeholders to catch any secondary variables
        for key, value_list in variables.items():
            placeholder_str = f"{{{key}}}"
            if placeholder_str in prompt:
                prompt = prompt.replace(placeholder_str, random.choice(value_list))
        
        # Cleanup: Remove curly braces if any remain (fallback)
        # prompt = prompt.replace("{", "").replace("}", "")
        
        data.append({
            "id": i + 1,
            "category": template["category"],
            "image_context": selection_context,
            "prompt": prompt
        })
    return data

# Generate 20k
large_dataset = generate_large_dataset(20000)
df_large = pd.DataFrame(large_dataset)
df_large.to_csv("multimodal_good_prompts_20k.csv", index=False)

print(f"Generated {len(df_large)} records.")
print(df_large["category"].value_counts())