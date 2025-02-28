import streamlit as st
import folium
import pandas as pd
import json
import geopandas as gpd
from unidecode import unidecode
from streamlit_folium import st_folium
from functools import partial
from folium.plugins import MousePosition
from branca.colormap import linear
from shapely.geometry import mapping

# ✅ Step 1: Cached Data Loading Functions
@st.cache_data
def load_geo_data():
    """Loads geographic data (uf_geometries and imm_geometries) from the data/ directory and computes bounding box."""
    uf_geometries = gpd.read_file("data/uf_geometries_simplified.geojson")
    imm_geometries = gpd.read_file("data/imm_geometries_simplified.geojson")

    # Compute bounding box
    coords = [
        tuple(c) for feature in json.loads(imm_geometries.to_json())["features"]
        for geom in [feature["geometry"]]
        if geom and geom["type"] in ["Polygon", "MultiPolygon"]
        for c in (geom["coordinates"][0] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]])
    ]

    # Extract min/max latitudes and longitudes
    lats = [c[1] for c in coords if isinstance(c, tuple) and len(c) == 2]
    lons = [c[0] for c in coords if isinstance(c, tuple) and len(c) == 2]

    # Default bounding box if coordinates are invalid
    bounds = [[-15, -55], [-10, -50]] if not lats or not lons else [[min(lats), min(lons)], [max(lats), max(lons)]]

    return uf_geometries, imm_geometries, bounds

@st.cache_data
def load_name_data():
    """Loads name encoding and counts from the data/ directory."""
    word_encoding_df = pd.read_csv("data/word_encoding.csv")
    word_counts_imm = pd.read_csv("data/word_counts_imm.csv")
    
    return word_encoding_df, word_counts_imm

# ✅ Load cached data
uf_geometries, imm_geometries, cached_bounds = load_geo_data()
word_encoding_df, word_counts_imm = load_name_data()

# ✅ Step 2: Compute ratios
def compute_ratio(words, _type):
    """Computes relative frequency of names in each municipality."""
    if isinstance(words, str):  
        words = [words]  

    encoded_words = word_encoding_df.loc[word_encoding_df['word'].isin(words), 'encoding']
    if encoded_words.empty:
        return None

    filtered_counts = word_counts_imm[word_counts_imm['encoded_word'].isin(encoded_words)]
    aggregated_counts = filtered_counts.groupby("COD_IMM")[_type].sum().reset_index()

    merged_data = imm_geometries.copy()
    merged_data = pd.merge(merged_data, aggregated_counts, on='COD_IMM', how='left').fillna(0)
    merged_data['ratio'] = (merged_data[_type] / merged_data['counts']) * 100
    merged_data['ratio'] = merged_data['ratio'].fillna(0)

    return merged_data

imm_geometries.to_file(f'imm_geometries.geojson', driver='GeoJSON')

# ✅ Step 3: Compute ratio difference
def compute_ratio_difference(words1, words2, _type, normalize=False):
    """Computes difference in ratios between two name groups."""
    ratio_data1 = compute_ratio(words1, _type)
    ratio_data2 = compute_ratio(words2, _type)

    # ✅ Handle cases where one or both groups return None
    if ratio_data1 is None and ratio_data2 is None:
        st.warning("⚠️ Nenhum dos nomes inseridos foi encontrado.")
        return None
    elif ratio_data1 is None:
        st.warning(f"⚠️ Nenhum dado encontrado para {words1}. Exibindo apenas {words2}.")
        ratio_data2["ratio_diff"] = ratio_data2["ratio"]  # ✅ Ensure valid column
        return ratio_data2
    elif ratio_data2 is None:
        st.warning(f"⚠️ Nenhum dado encontrado para {words2}. Exibindo apenas {words1}.")
        ratio_data1["ratio_diff"] = ratio_data1["ratio"]  # ✅ Ensure valid column
        return ratio_data1

    # ✅ Merge data safely
    merged = pd.merge(
        ratio_data1[['COD_IMM', 'ratio']], 
        ratio_data2[['COD_IMM', 'ratio']], 
        on='COD_IMM', suffixes=('_group1', '_group2'), how='outer'
    ).fillna(0)

    if normalize:
        max1, max2 = merged['ratio_group1'].max(), merged['ratio_group2'].max()
        if max1 > 0: merged['ratio_group1'] /= max1
        if max2 > 0: merged['ratio_group2'] /= max2
        merged[['ratio_group1', 'ratio_group2']] *= 100

    merged['ratio_diff'] = merged['ratio_group1'] - merged['ratio_group2']
    final_data = imm_geometries.copy()
    return pd.merge(final_data, merged, on='COD_IMM', how='left').fillna(0)


# ✅ Step 4: Define colormaps and styles
def get_comparison_colormap(min_diff, max_diff):
    """Returns distinct colormap for ratio difference visualization."""
    largest = (max(abs(min_diff), abs(max_diff)))
    return linear.RdBu_10.scale(-largest, largest)  

def style_function(feature, cmap, column):
    """Applies dynamic coloring."""
    return {'fillColor': cmap(feature['properties'][column]), 'fillOpacity': 0.9, 'color': 'black', 'weight': 0.7, 'opacity': 0.5}

def style_function_int(feature):
    """State boundaries style."""
    return {'fillOpacity': 0, 'color': 'black', 'weight': 1}

# ✅ Step 5: Generate the Map
def generate_map(words, _type):
    """Generates frequency choropleth map."""
    ratio_data = compute_ratio(words, _type)
    if ratio_data is None:
        return None

    cmap = linear.YlOrRd_04.scale(ratio_data['ratio'].min(), ratio_data['ratio'].max())

    choromap = folium.Map(location=[(cached_bounds[0][0] + cached_bounds[1][0]) / 2, 
                                    (cached_bounds[0][1] + cached_bounds[1][1]) / 2], 
                          zoom_start=6, max_bounds=cached_bounds)

    choromap.fit_bounds(cached_bounds)

    folium.GeoJson(ratio_data, style_function=partial(style_function, cmap=cmap, column='ratio'), 
                   tooltip=folium.GeoJsonTooltip(fields=['IMM', 'ratio', 'counts'], aliases=['Região', 'Frequência (%)', 'Amostra'], localize=True)).add_to(choromap)
    folium.GeoJson(uf_geometries, style_function=style_function_int, interactive=False).add_to(choromap)
    cmap.add_to(choromap)

    return choromap

def generate_difference_map(words1, words2, _type, normalize=False):
    """Generates difference map between two name groups."""
    ratio_diff_data = compute_ratio_difference(words1, words2, _type, normalize)

    min_diff, max_diff = ratio_diff_data['ratio_diff'].min(), ratio_diff_data['ratio_diff'].max()
    cmap = get_comparison_colormap(min_diff, max_diff)

    choromap = folium.Map(location=[(cached_bounds[0][0] + cached_bounds[1][0]) / 2, 
                                    (cached_bounds[0][1] + cached_bounds[1][1]) / 2], 
                          zoom_start=6, max_bounds=cached_bounds)

    choromap.fit_bounds(cached_bounds)

    folium.GeoJson(ratio_diff_data, style_function=partial(style_function, cmap=cmap, column='ratio_diff'), 
                   tooltip=folium.GeoJsonTooltip(fields=['IMM', 'ratio_group1', 'ratio_group2', 'ratio_diff'], 
                                                 aliases=['Região', 'Freq Grupo 1 (%)', 'Freq Grupo 2 (%)', 'Diferença (%)'], localize=True)).add_to(choromap)
    folium.GeoJson(uf_geometries, style_function=style_function_int, interactive=False).add_to(choromap)
    cmap.add_to(choromap)

    return choromap

# ✅ Step 6: Build Streamlit UI
st.title("Distribuição de Nomes no Brasil")

# ✅ Store user input in session state to persist selections
if "map_generated" not in st.session_state:
    st.session_state["map_generated"] = True
if "selected_analysis" not in st.session_state:
    st.session_state["selected_analysis"] = "Frequência"
if "selected_words" not in st.session_state:
    st.session_state["selected_words"] = []
if "selected_words1" not in st.session_state:
    st.session_state["selected_words1"] = []
if "selected_words2" not in st.session_state:
    st.session_state["selected_words2"] = []
if "selected_count_type" not in st.session_state:
    st.session_state["selected_count_type"] = "sobre_count"
if "selected_normalize" not in st.session_state:
    st.session_state["selected_normalize"] = False

# ✅ Selection for frequency vs comparison
selected_analysis = st.radio("Escolha o tipo de análise:", ["Frequência", "Comparação"])

count_type_key = {
    'Sobrenome': 'sobre_count',
    'Prenome': 'pre_count'
}

# ✅ Handle frequency analysis
if selected_analysis == "Frequência":
    words = unidecode(st.text_input("Digite um nome ou lista de nomes (separados por vírgula):", value='SOUZA, SOUSA')).upper().split(",")
    count_type = st.selectbox("Escolha o tipo de contagem:", ['Sobrenome', 'Prenome'])

    sorted_words = sorted(set(words))  # ✅ Sort names to detect changes

    if (sorted_words != st.session_state["selected_words"] or count_type != st.session_state["selected_count_type"]) and (sorted_words and sorted_words[0]):
        st.session_state["selected_words"] = sorted_words
        st.session_state["selected_count_type"] = count_type_key[count_type]
        st.session_state["map_generated"] = True  # Trigger map update

    if st.session_state["map_generated"]:
        with st.spinner("🔄 Gerando mapa..."):
            choromap = generate_map(st.session_state["selected_words"], st.session_state["selected_count_type"])
            st_folium(choromap, use_container_width=True, height=600, returned_objects=[])


# ✅ Handle comparison analysis
if selected_analysis == "Comparação":
    words1 = unidecode(st.text_input("Grupo 1:", value='SOUZA')).upper().split(",")
    words2 = unidecode(st.text_input("Grupo 2:", value='SOUSA')).upper().split(",")
    count_type = st.selectbox("Escolha o tipo de contagem:", ['Sobrenome', 'Prenome'])
    normalize = st.checkbox("Normalizar?")

    sorted_words1 = sorted(set(words1))  # ✅ Sort names to detect changes
    sorted_words2 = sorted(set(words2))  # ✅ Sort names to detect changes

    if (
        (sorted_words1 != st.session_state["selected_words1"]
        or sorted_words2 != st.session_state["selected_words2"]
        or normalize != st.session_state["selected_normalize"])
        and (sorted_words1 and sorted_words1[0] and sorted_words2 and sorted_words2[0])
    ):
        st.session_state["selected_words1"] = sorted_words1
        st.session_state["selected_words2"] = sorted_words2
        st.session_state["selected_normalize"] = normalize
        st.session_state["selected_count_type"] = count_type_key[count_type]
        st.session_state["map_generated"] = True  # Trigger map update

    if st.session_state["map_generated"]:
        with st.spinner("🔄 Gerando mapa..."):
            choromap = generate_difference_map(
                st.session_state["selected_words1"], 
                st.session_state["selected_words2"], 
                st.session_state["selected_count_type"], 
                st.session_state["selected_normalize"]
            )
            st_folium(choromap, use_container_width=True, height=600, returned_objects=[])
