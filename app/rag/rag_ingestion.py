"""VoyageAI travel knowledge-base builder.

Converted from the uploaded notebook-style code into an import-safe Python module.

Usage from terminal:
    python travel_knowledge_base.py

Usage from another file:
    from travel_knowledge_base import build_vector_store, retrieve_travel_info

    vector_store = build_vector_store(reset_db=True)
    results = retrieve_travel_info("beaches and seafood", vector_store, k=3)
"""

from pathlib import Path
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


COLLECTION_NAME = "voyageai_travel_knowledge"


TRAVEL_DOCS = {
    "goa.txt": """
Destination: Goa
State/Region: Goa
Destination Type: Beach, nightlife, seafood, coastal vacation, friends trip, couples trip

Overview:
Goa is one of India's most popular beach destinations. It is known for beaches, nightlife, seafood, Portuguese heritage, forts, churches, markets, and relaxed coastal culture.

Best For:
- Beach lovers
- Seafood lovers
- Friends groups
- Couples
- Nightlife travelers
- Relaxed vacation seekers

Top Attractions:
- Baga Beach
- Calangute Beach
- Anjuna Beach
- Vagator Beach
- Fort Aguada
- Chapora Fort
- Basilica of Bom Jesus
- Panjim
- Dudhsagar Falls

Food Recommendations:
- Goan fish curry
- Prawn balchao
- Pork vindaloo
- Xacuti
- Bebinca
- Seafood thali

Stay Suggestions:
- Budget: hostels near Anjuna, Vagator, or Baga
- Comfort: boutique hotels near Candolim or Panjim
- Luxury: beach resorts in South Goa

Best Time to Visit:
November to February is the most comfortable and popular season. Monsoon is scenic but beach activities may be limited.

Ideal Duration:
3 to 5 days

Travel Tips:
- Rent a scooter for local travel if comfortable.
- North Goa is better for nightlife.
- South Goa is better for peaceful beaches and luxury stays.
""",

    "jaipur.txt": """
Destination: Jaipur
State/Region: Rajasthan
Destination Type: Heritage, forts, palaces, culture, food, family trip

Overview:
Jaipur, also known as the Pink City, is famous for royal palaces, forts, colorful markets, traditional Rajasthani food, and heritage architecture.

Best For:
- History lovers
- Culture lovers
- Families
- Architecture enthusiasts
- Photographers
- First-time Rajasthan travelers

Top Attractions:
- Amber Fort
- City Palace
- Hawa Mahal
- Jantar Mantar
- Nahargarh Fort
- Jaigarh Fort
- Jal Mahal
- Albert Hall Museum
- Patrika Gate

Food Recommendations:
- Dal baati churma
- Laal maas
- Ghewar
- Pyaaz kachori
- Mirchi vada
- Gatte ki sabzi
- Rajasthani thali

Stay Suggestions:
- Budget: stays near MI Road or Bani Park
- Comfort: heritage-style hotels
- Luxury: palace hotels and premium heritage resorts

Best Time to Visit:
October to March is best because the weather is more suitable for sightseeing.

Ideal Duration:
2 to 4 days

Travel Tips:
- Start sightseeing early to avoid crowds.
- Keep time for local markets like Johari Bazaar and Bapu Bazaar.
""",

    "manali.txt": """
Destination: Manali
State/Region: Himachal Pradesh
Destination Type: Mountains, snow, adventure, honeymoon, backpacking, cafes

Overview:
Manali is a popular hill station known for mountains, snow activities, adventure sports, riverside stays, cafes, and scenic valleys.

Best For:
- Adventure seekers
- Honeymoon couples
- Backpackers
- Nature lovers
- Bikers
- Snow lovers

Top Attractions:
- Solang Valley
- Hadimba Temple
- Old Manali
- Manu Temple
- Mall Road
- Vashisht Hot Springs
- Atal Tunnel
- Sissu
- Rohtang Pass

Food Recommendations:
- Trout fish
- Siddu
- Thukpa
- Momos
- Himachali dham
- Cafe-style continental food in Old Manali

Stay Suggestions:
- Budget: hostels in Old Manali
- Comfort: riverside cottages
- Luxury: mountain resorts with valley views

Best Time to Visit:
March to June is good for pleasant weather. December to February is good for snow experiences.

Ideal Duration:
4 to 6 days

Travel Tips:
- Rohtang Pass access depends on weather and government rules.
- Old Manali is better for cafes and backpacker stays.
""",

    "kerala.txt": """
Destination: Kerala
State/Region: Kerala
Destination Type: Backwaters, beaches, Ayurveda, nature, wellness, slow travel

Overview:
Kerala is known for backwaters, beaches, tea gardens, Ayurveda, houseboats, wildlife, tropical greenery, and peaceful travel experiences.

Best For:
- Families
- Couples
- Nature lovers
- Wellness travelers
- Food lovers
- Slow travel seekers

Top Attractions:
- Munnar
- Alleppey
- Kochi
- Thekkady
- Wayanad
- Varkala
- Kovalam
- Kumarakom
- Athirapally Falls

Food Recommendations:
- Appam with stew
- Kerala sadya
- Puttu and kadala curry
- Malabar biryani
- Fish molee
- Banana chips
- Seafood

Stay Suggestions:
- Budget: homestays and hostels
- Comfort: boutique resorts and houseboats
- Luxury: Ayurveda resorts and private pool villas

Best Time to Visit:
October to March is best for general travel. Monsoon is good for greenery and Ayurveda-focused trips.

Ideal Duration:
5 to 8 days

Travel Tips:
- Combine Munnar, Alleppey, and Kochi for a balanced first trip.
- Houseboat stays are best planned in advance.
""",

    "andaman.txt": """
Destination: Andaman and Nicobar Islands
State/Region: Union Territory
Destination Type: Islands, beaches, scuba diving, snorkeling, honeymoon, water sports

Overview:
Andaman is known for clean beaches, blue waters, coral reefs, scuba diving, snorkeling, island hopping, and peaceful tropical experiences.

Best For:
- Beach lovers
- Honeymoon couples
- Scuba diving beginners
- Snorkeling lovers
- Nature lovers
- Luxury island travelers

Top Attractions:
- Radhanagar Beach
- Havelock Island
- Neil Island
- Cellular Jail
- North Bay Island
- Ross Island
- Elephant Beach
- Bharatpur Beach

Food Recommendations:
- Seafood
- Fish curry
- Prawn dishes
- Crab dishes
- Coconut-based coastal food
- Indian and continental island cafes

Stay Suggestions:
- Budget: simple guesthouses in Port Blair or Havelock
- Comfort: beachside cottages
- Luxury: premium island resorts

Best Time to Visit:
October to May is generally suitable for beach and water activities.

Ideal Duration:
5 to 7 days

Travel Tips:
- Keep buffer time for ferry transfers.
- Book water activities in advance during peak season.
""",

    "udaipur.txt": """
Destination: Udaipur
State/Region: Rajasthan
Destination Type: Lakes, palaces, romantic trip, heritage, luxury, culture

Overview:
Udaipur is known as the City of Lakes. It is famous for lake views, royal palaces, heritage hotels, romantic boat rides, rooftop cafes, and cultural experiences.

Best For:
- Couples
- Luxury travelers
- Heritage lovers
- Photographers
- Family travelers
- Slow travel seekers

Top Attractions:
- City Palace
- Lake Pichola
- Jag Mandir
- Fateh Sagar Lake
- Sajjangarh Monsoon Palace
- Saheliyon Ki Bari
- Bagore Ki Haveli
- Jagdish Temple

Food Recommendations:
- Dal baati churma
- Rajasthani thali
- Laal maas
- Gatte ki sabzi
- Kachori
- Rooftop cafe meals near lakes

Stay Suggestions:
- Budget: guesthouses near old city
- Comfort: lake-view boutique hotels
- Luxury: palace hotels and lake resorts

Best Time to Visit:
October to March is best for pleasant sightseeing.

Ideal Duration:
2 to 4 days

Travel Tips:
- Choose a lake-view stay for a better experience.
- Sunset near Lake Pichola is highly recommended.
""",

    "rishikesh.txt": """
Destination: Rishikesh
State/Region: Uttarakhand
Destination Type: Adventure, yoga, spirituality, river rafting, backpacking

Overview:
Rishikesh is known for river rafting, yoga retreats, Ganga ghats, cafes, temples, adventure activities, and peaceful Himalayan surroundings.

Best For:
- Adventure seekers
- Yoga and wellness travelers
- Backpackers
- Spiritual travelers
- Friends groups
- Budget travelers

Top Attractions:
- Lakshman Jhula
- Ram Jhula
- Triveni Ghat
- Beatles Ashram
- Neer Garh Waterfall
- River rafting points
- Parmarth Niketan
- Ganga Aarti

Food Recommendations:
- North Indian vegetarian food
- Israeli cafe food
- Smoothie bowls
- Chai
- Satvik meals
- Street snacks

Stay Suggestions:
- Budget: hostels near Tapovan
- Comfort: riverside camps or guesthouses
- Luxury: wellness resorts near the Ganga

Best Time to Visit:
September to June is commonly preferred. Rafting availability depends on season and local rules.

Ideal Duration:
2 to 4 days

Travel Tips:
- Rishikesh is mostly vegetarian and alcohol-free in many areas.
- Adventure activities should be booked through licensed operators.
""",

    "varanasi.txt": """
Destination: Varanasi
State/Region: Uttar Pradesh
Destination Type: Spiritual, culture, heritage, ghats, temples, photography

Overview:
Varanasi is one of India's oldest and most spiritual cities. It is known for ghats, temples, Ganga Aarti, narrow lanes, classical culture, and deep religious significance.

Best For:
- Spiritual travelers
- Culture lovers
- Photographers
- History lovers
- Solo travelers
- Slow travel seekers

Top Attractions:
- Dashashwamedh Ghat
- Assi Ghat
- Kashi Vishwanath Temple
- Manikarnika Ghat
- Sarnath
- Banaras Hindu University
- Ramnagar Fort
- Ganga boat ride

Food Recommendations:
- Kachori sabzi
- Tamatar chaat
- Banarasi paan
- Lassi
- Malaiyyo
- Thandai
- Chaat

Stay Suggestions:
- Budget: guesthouses near Assi Ghat
- Comfort: heritage stays near ghats
- Luxury: river-view hotels

Best Time to Visit:
October to March is best for comfortable weather.

Ideal Duration:
2 to 3 days

Travel Tips:
- Attend morning boat ride and evening Ganga Aarti.
- Respect local customs around temples and ghats.
""",

    "ladakh.txt": """
Destination: Ladakh
State/Region: Union Territory of Ladakh
Destination Type: Mountains, road trip, adventure, high altitude, biking, landscapes

Overview:
Ladakh is known for high-altitude landscapes, monasteries, mountain passes, lakes, road trips, biking routes, and dramatic cold desert scenery.

Best For:
- Adventure travelers
- Bikers
- Road trip lovers
- Landscape photographers
- Mountain lovers
- Experienced travelers

Top Attractions:
- Leh
- Pangong Lake
- Nubra Valley
- Magnetic Hill
- Khardung La
- Shanti Stupa
- Thiksey Monastery
- Hemis Monastery
- Tso Moriri

Food Recommendations:
- Thukpa
- Momos
- Skyu
- Tingmo
- Butter tea
- Ladakhi bread
- Tibetan-style meals

Stay Suggestions:
- Budget: guesthouses in Leh
- Comfort: camps in Nubra or Pangong
- Luxury: premium mountain stays in Leh

Best Time to Visit:
May to September is commonly preferred for road access and sightseeing.

Ideal Duration:
6 to 9 days

Travel Tips:
- Acclimatization is very important due to high altitude.
- Keep the first day light after arrival in Leh.
""",

    "kashmir.txt": """
Destination: Kashmir
State/Region: Jammu and Kashmir
Destination Type: Mountains, valleys, snow, honeymoon, family trip, scenic vacation

Overview:
Kashmir is known for scenic valleys, snow-capped mountains, gardens, lakes, houseboats, meadows, and romantic landscapes.

Best For:
- Honeymoon couples
- Families
- Nature lovers
- Snow lovers
- Photographers
- Relaxed scenic travelers

Top Attractions:
- Srinagar
- Dal Lake
- Gulmarg
- Pahalgam
- Sonamarg
- Mughal Gardens
- Shankaracharya Temple
- Betaab Valley

Food Recommendations:
- Rogan josh
- Yakhni
- Dum aloo
- Kahwa
- Wazwan
- Kashmiri pulao
- Sheer chai

Stay Suggestions:
- Budget: guesthouses in Srinagar
- Comfort: houseboats or hotels near Dal Lake
- Luxury: premium resorts in Gulmarg or Pahalgam

Best Time to Visit:
March to October is good for valleys and sightseeing. December to February is good for snow experiences.

Ideal Duration:
5 to 7 days

Travel Tips:
- Keep extra time for weather-related changes.
- Houseboat stays are a unique Kashmir experience.
""",

    "darjeeling.txt": """
Destination: Darjeeling
State/Region: West Bengal
Destination Type: Hills, tea gardens, toy train, family trip, peaceful vacation

Overview:
Darjeeling is a hill station known for tea gardens, views of Kanchenjunga, colonial charm, toy train rides, monasteries, and cool weather.

Best For:
- Families
- Couples
- Tea lovers
- Hill station travelers
- Photographers
- Slow travelers

Top Attractions:
- Tiger Hill
- Darjeeling Himalayan Railway
- Batasia Loop
- Peace Pagoda
- Himalayan Mountaineering Institute
- Padmaja Naidu Himalayan Zoological Park
- Tea gardens
- Mall Road

Food Recommendations:
- Momos
- Thukpa
- Darjeeling tea
- Nepali thali
- Tibetan bread
- Noodles
- Local bakery items

Stay Suggestions:
- Budget: homestays and guesthouses
- Comfort: hill-view hotels
- Luxury: colonial-style premium stays

Best Time to Visit:
March to May and October to December are commonly preferred.

Ideal Duration:
3 to 4 days

Travel Tips:
- Wake up early for Tiger Hill sunrise.
- Weather can change quickly, so carry warm layers.
""",

    "pondicherry.txt": """
Destination: Pondicherry
State/Region: Puducherry
Destination Type: Beaches, French colony, cafes, spiritual, weekend trip

Overview:
Pondicherry is known for French-style streets, beaches, cafes, Auroville, spiritual experiences, and relaxed coastal charm.

Best For:
- Couples
- Weekend travelers
- Cafe lovers
- Beach lovers
- Solo travelers
- Photography lovers

Top Attractions:
- White Town
- Promenade Beach
- Paradise Beach
- Auroville
- Sri Aurobindo Ashram
- Rock Beach
- Serenity Beach
- French cafes

Food Recommendations:
- French bakery items
- Crepes
- Seafood
- South Indian meals
- Continental cafe food
- Filter coffee
- Gelato

Stay Suggestions:
- Budget: guesthouses near Heritage Town
- Comfort: boutique stays in White Town
- Luxury: beach resorts near outskirts

Best Time to Visit:
October to March is best for pleasant weather.

Ideal Duration:
2 to 3 days

Travel Tips:
- Rent a bicycle or scooter to explore White Town.
- Keep one day for Auroville and beach exploration.
""",

    "agra.txt": """
Destination: Agra
State/Region: Uttar Pradesh
Destination Type: Heritage, monuments, history, architecture, weekend trip

Overview:
Agra is famous for the Taj Mahal and Mughal heritage. It is a popular destination for history, architecture, photography, and short trips.

Best For:
- History lovers
- Architecture lovers
- Families
- Couples
- Photographers
- First-time India travelers

Top Attractions:
- Taj Mahal
- Agra Fort
- Mehtab Bagh
- Fatehpur Sikri
- Itmad-ud-Daulah
- Akbar's Tomb
- Kinari Bazaar

Food Recommendations:
- Agra petha
- Mughlai food
- Bedai and jalebi
- Paratha
- Chaat
- Kebab
- North Indian thali

Stay Suggestions:
- Budget: hotels near railway station or Taj Ganj
- Comfort: mid-range hotels near Taj Mahal
- Luxury: premium hotels with Taj views

Best Time to Visit:
October to March is best for sightseeing.

Ideal Duration:
1 to 2 days

Travel Tips:
- Visit Taj Mahal early morning for better light and fewer crowds.
- Agra can be combined with Delhi and Jaipur as part of the Golden Triangle.
""",

    "delhi.txt": """
Destination: Delhi
State/Region: Delhi NCR
Destination Type: History, food, shopping, monuments, city experience

Overview:
Delhi is India's capital city and offers a mix of Mughal heritage, colonial architecture, markets, street food, museums, and modern urban experiences.

Best For:
- History lovers
- Food lovers
- Shoppers
- Families
- First-time India travelers
- Museum lovers

Top Attractions:
- Red Fort
- India Gate
- Qutub Minar
- Humayun's Tomb
- Lotus Temple
- Akshardham
- Connaught Place
- Chandni Chowk
- National Museum

Food Recommendations:
- Chole bhature
- Parathas
- Butter chicken
- Chaat
- Kebabs
- Momos
- Jalebi
- Street food in Chandni Chowk

Stay Suggestions:
- Budget: Paharganj or Karol Bagh
- Comfort: Connaught Place or South Delhi hotels
- Luxury: premium hotels in Aerocity, Central Delhi, or South Delhi

Best Time to Visit:
October to March is better for outdoor sightseeing.

Ideal Duration:
2 to 4 days

Travel Tips:
- Use Delhi Metro for efficient local travel.
- Keep extra travel time because traffic can be heavy.
""",

    "mumbai.txt": """
Destination: Mumbai
State/Region: Maharashtra
Destination Type: City life, beaches, food, Bollywood, nightlife, business travel

Overview:
Mumbai is India's financial and entertainment hub. It is known for sea-facing promenades, street food, colonial architecture, Bollywood culture, nightlife, and fast-paced city life.

Best For:
- City explorers
- Food lovers
- Nightlife travelers
- Bollywood fans
- Business travelers
- Weekend travelers

Top Attractions:
- Gateway of India
- Marine Drive
- Colaba Causeway
- Elephanta Caves
- Juhu Beach
- Bandra Bandstand
- Chhatrapati Shivaji Maharaj Terminus
- Siddhivinayak Temple
- Haji Ali Dargah

Food Recommendations:
- Vada pav
- Pav bhaji
- Bhel puri
- Bombay sandwich
- Misal pav
- Seafood
- Irani cafe food

Stay Suggestions:
- Budget: hostels or budget hotels in Andheri or Colaba
- Comfort: hotels in Bandra, Andheri, or South Mumbai
- Luxury: sea-facing hotels in South Mumbai or Juhu

Best Time to Visit:
November to February is more comfortable for city exploration.

Ideal Duration:
2 to 4 days

Travel Tips:
- Use local trains only if comfortable with crowds.
- Marine Drive is best during evening.
"""
}


def get_project_root() -> Path:
    """Return the project root whether this file runs from notebooks/, agents/, or root."""
    current_dir = Path.cwd()

    if current_dir.name == "agents":
        return current_dir.parent.parent
    if current_dir.name == "notebooks":
        return current_dir.parent
    return current_dir


def get_project_paths(project_root: str | Path | None = None) -> tuple[Path, Path, Path]:
    """Return project root, travel docs directory, and Chroma DB directory."""
    root = Path(project_root) if project_root else get_project_root()
    data_dir = root / "data" / "travel_docs"
    chroma_dir = root / "chroma_db"
    return root, data_dir, chroma_dir


def create_travel_documents(
    data_dir: str | Path | None = None,
    project_root: str | Path | None = None,
    overwrite: bool = True,
) -> list[Path]:
    """Create destination text files used as the RAG knowledge source."""
    _, default_data_dir, _ = get_project_paths(project_root)
    data_dir = Path(data_dir) if data_dir else default_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in TRAVEL_DOCS.items():
        file_path = data_dir / filename
        if overwrite or not file_path.exists():
            file_path.write_text(content.strip(), encoding="utf-8")

    return sorted(data_dir.glob("*.txt"))


def load_travel_documents(
    data_dir: str | Path | None = None,
    project_root: str | Path | None = None,
):
    """Load destination text files as LangChain documents with metadata."""
    _, default_data_dir, _ = get_project_paths(project_root)
    data_dir = Path(data_dir) if data_dir else default_data_dir

    documents = []
    for file_path in sorted(data_dir.glob("*.txt")):
        loader = TextLoader(str(file_path), encoding="utf-8")
        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.metadata["source"] = file_path.name
            doc.metadata["city"] = file_path.stem

        documents.extend(loaded_docs)

    return documents


def split_travel_documents(
    documents,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
):
    """Split loaded documents into overlapping chunks for vector search."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return text_splitter.split_documents(documents)


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
):
    """Create the HuggingFace embedding model used by Chroma."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    reset_db: bool = True,
    project_root: str | Path | None = None,
    data_dir: str | Path | None = None,
    chroma_dir: str | Path | None = None,
    collection_name: str = COLLECTION_NAME,
):
    """Create travel docs, split them, embed them, and store them in Chroma."""
    root, default_data_dir, default_chroma_dir = get_project_paths(project_root)
    data_dir = Path(data_dir) if data_dir else default_data_dir
    chroma_dir = Path(chroma_dir) if chroma_dir else default_chroma_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    create_travel_documents(data_dir=data_dir, overwrite=True)
    documents = load_travel_documents(data_dir=data_dir)
    chunks = split_travel_documents(documents)
    embedding_model = get_embedding_model()

    if reset_db and chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(chroma_dir),
        collection_name=collection_name,
    )

    print("Project root:", root)
    print("Travel docs folder:", data_dir)
    print("Chroma DB folder:", chroma_dir)
    print(f"Total documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")
    print("Chroma vector database created successfully.")
    print("Total vectors stored:", vector_store._collection.count())

    return vector_store


def load_vector_store(
    project_root: str | Path | None = None,
    chroma_dir: str | Path | None = None,
    collection_name: str = COLLECTION_NAME,
):
    """Load an existing Chroma vector store without rebuilding it."""
    _, _, default_chroma_dir = get_project_paths(project_root)
    chroma_dir = Path(chroma_dir) if chroma_dir else default_chroma_dir
    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=embedding_model,
        collection_name=collection_name,
    )


def retrieve_travel_info(
    query: str,
    vector_store=None,
    k: int = 3,
    project_root: str | Path | None = None,
    print_results: bool = True,
):
    """Retrieve matching travel knowledge chunks for a user query."""
    if vector_store is None:
        vector_store = load_vector_store(project_root=project_root)

    results = vector_store.similarity_search_with_score(query, k=k)

    if print_results:
        print(f"Query: {query}")
        print("=" * 80)

        for i, (doc, score) in enumerate(results, start=1):
            print(f"\nResult {i}")
            print(f"Similarity Score: {score}")
            print(f"Source: {doc.metadata.get('source')}")
            print(f"City: {doc.metadata.get('city')}")
            print("-" * 80)
            print(doc.page_content)

    return results


def main() -> None:
    """Run the same test flow that was present in the notebook."""
    vector_store = build_vector_store(reset_db=True)

    retrieve_travel_info(
        "Which destination is good for beaches and seafood?",
        vector_store=vector_store,
        k=2,
    )
    retrieve_travel_info(
        "Suggest a place for snow and adventure activities",
        vector_store=vector_store,
        k=2,
    )
    retrieve_travel_info(
        "Which place is best for backwaters and Ayurveda?",
        vector_store=vector_store,
    )


if __name__ == "__main__":
    main()
