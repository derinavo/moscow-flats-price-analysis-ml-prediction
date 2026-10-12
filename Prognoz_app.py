from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Прогноз стоимости недвижимости",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "best2.pkl"
DATA_PATH = ROOT / "realty_data.csv"


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    flats = df[df["city"] == "Москва"].copy()

    flats = flats[
        flats["price"].notna() &
        flats["total_square"].notna() &
        (flats["price"] > 0) &
        (flats["total_square"] > 0)
    ].copy()

    flats["price_mln"] = flats["price"] / 1_000_000
    flats["price_per_sqm"] = flats["price"] / flats["total_square"]
    flats["price_per_sqm_th"] = flats["price_per_sqm"] / 1000

    red_square_lat = 55.75417
    red_square_lon = 37.62000

    lat = np.radians(flats["lat"])
    lon = np.radians(flats["lon"])
    center_lat = np.radians(red_square_lat)
    center_lon = np.radians(red_square_lon)

    dlat = lat - center_lat
    dlon = lon - center_lon

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(center_lat) * np.cos(lat) * np.sin(dlon / 2) ** 2
    )

    flats["distance_to_center_km"] = (
        2 * 6371 * np.arcsin(np.sqrt(a))
    )

    return flats


def get_distance_zona(distance):
    bins = [0, 3, 5, 10, 15, 20, 30, np.inf]

    labels = [
        "0–3 км",
        "3–5 км",
        "5–10 км",
        "10–15 км",
        "15–20 км",
        "20–30 км",
        "30+ км"
    ]

    return pd.cut(
        pd.Series([distance]),
        bins=bins,
        labels=labels,
        right=False
    ).astype("object").iloc[0]

bundle = load_bundle()

model = bundle["model"]
metadata = bundle["metadata"]

flats = load_data()

model_districts = set(metadata["districts"])

csv_districts = set(
    flats["district"]
    .dropna()
    .astype(str)
    .unique()
)

district_options = sorted(
    model_districts.intersection(csv_districts)
)

#медианное расстояние до центра для каждого района
district_distance = (
    flats.dropna(
        subset=[
            "district",
            "distance_to_center_km"
        ]
    )
    .assign(
        district=lambda x: x["district"].astype(str)
    )
    .groupby("district")["distance_to_center_km"]
    .median()
    .to_dict()
)


#скрытый признак, не удалять!
available_sources = set(metadata["sources"])

source_counts = (
    flats["source"]
    .dropna()
    .astype(str)
    .value_counts()
)

default_source = next(
    (
        source
        for source in source_counts.index
        if source in available_sources
    ),
    metadata["sources"][0]
)


#пподбираем район, медианное расстояние которого ближе всего к введённому
district_distance_valid = {
    district: distance
    for district, distance in district_distance.items()
    if district in model_districts
    and pd.notna(distance)
}


def infer_district_from_distance(distance):
    if not district_distance_valid:
        return district_options[0]

    return min(
        district_distance_valid,
        key=lambda district: abs(
            district_distance_valid[district] - distance
        )
    )

st.title("Анализ и прогноз стоимости квартир в Москве")

tab1, tab2 = st.tabs(
    [
        "Обзор рынка",
        "Прогноз цены"
    ]
)

with tab1:

    st.subheader("Обзор рынка")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Количество объектов",
        f"{len(flats):,}".replace(",", " ")
    )

    col2.metric(
        "Медианная цена",
        f"{flats['price_mln'].median():.1f} млн ₽"
    )

    col3.metric(
        "Медианная цена за м2",
        f"{flats['price_per_sqm_th'].median():.0f} тыс. ₽"
    )

    col4.metric(
        "Медианная площадь",
        f"{flats['total_square'].median():.0f} м²"
    )

    st.divider()

    st.subheader("Фильтры")

    f1, f2 = st.columns(2)

    selected_districts = f1.multiselect(
        "Район",
        options=district_options
    )

    room_values = sorted(
        flats["rooms"]
        .dropna()
        .unique()
    )

    selected_rooms = f2.multiselect(
        "Количество комнат",
        options=room_values
    )

    filtered = flats.copy()

    if selected_districts:
        filtered = filtered[
            filtered["district"]
            .astype(str)
            .isin(selected_districts)
        ]

    if selected_rooms:
        filtered = filtered[
            filtered["rooms"].isin(selected_rooms)
        ]

    st.write(
        (
            f"После фильтрации: "
            f"**{len(filtered):,} объектов**"
        ).replace(",", " ")
    )

    district_stats = (
        filtered
        .dropna(
            subset=[
                "district",
                "price_per_sqm_th"
            ]
        )
        .assign(
            district=lambda x: x["district"].astype(str)
        )
        .groupby("district")
        .agg(
            listings=("price", "size"),
            median_price_sqm_th=(
                "price_per_sqm_th",
                "median"
            )
        )
        .query("listings >= 20")
        .sort_values(
            "median_price_sqm_th",
            ascending=False
        )
        .head(15)
    )

    st.subheader(
        "Топ районов по медианной цене за м2"
    )

    if not district_stats.empty:
        st.bar_chart(
            district_stats[
                "median_price_sqm_th"
            ]
        )
    else:
        st.info(
            "Недостаточно данных после фильтрации"
        )

    st.subheader("Распределение цены")

    chart_data = filtered[
        ["price_mln"]
    ].dropna()

    if not chart_data.empty:

        q99 = chart_data[
            "price_mln"
        ].quantile(0.99)

        chart_data = chart_data[
            chart_data["price_mln"] <= q99
        ]

        st.bar_chart(
            chart_data[
                "price_mln"
            ]
            .round()
            .value_counts()
            .sort_index()
        )


with tab2:

    st.subheader(
        "Прогноз стоимости квартиры"
    )

    st.write(
        "Введите характеристики объекта"
        "Для географии выберите район или расстояние до Красной площади"
    )

    left, right = st.columns(2)

    with left:

        total_square = st.number_input(
            "Площадь, м²",
            min_value=10.0,
            max_value=1500.0,
            value=60.0,
            step=1.0
        )

        rooms = st.number_input(
            "Количество комнат",
            min_value=1,
            max_value=15,
            value=2,
            step=1
        )

    with right:

        floor = st.number_input(
            "Этаж",
            min_value=1,
            max_value=100,
            value=5,
            step=1
        )

    st.divider()

    geo_mode = st.radio(
        "Как указать местоположение?",
        options=[
            "Район",
            "Расстояние до Красной площади"
        ],
        horizontal=True
    )

    if geo_mode == "Район":

        district = st.selectbox(
            "Район",
            options=district_options
        )

        distance = float(
            district_distance.get(
                district,
                flats["distance_to_center_km"].median()
            )
        )

        st.caption(
            f"Для модели будет использовано медианное "
            f"расстояние района до центра: "
            f"{distance:.1f} км."
        )

    else:

        distance = st.number_input(
            "Расстояние до Красной площади, км",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5
        )

        district = infer_district_from_distance(
            float(distance)
        )

        st.caption(
            f"Для модели автоматически выбран ближайший "
            f"по медианному расстоянию район: "
            f"{district}."
        )

    is_first_floor = int(
        floor == 1
    )

    distance_zona = get_distance_zona(
        float(distance)
    )

    input_data = pd.DataFrame({
        "total_square": [
            float(total_square)
        ],
        "rooms": [
            float(rooms)
        ],
        "floor": [
            float(floor)
        ],
        "distance_to_center_km": [
            float(distance)
        ],
        "is_first_floor": [
            is_first_floor
        ],
        "district": [
            district
        ],
        "source": [
            default_source
        ],
        "distance_zona": [
            distance_zona
        ]
    })

    st.caption(
        f"Географическая зона: {distance_zona}. "
        f"Первый этаж: "
        f"{'да' if is_first_floor else 'нет'}."
    )


    if st.button(
        "Рассчитать стоимость",
        type="primary",
        use_container_width=True
    ):

        try:

            prediction = float(
                model.predict(
                    input_data
                )[0]
            )

            price_per_sqm = (
                prediction /
                total_square
            )

            st.divider()

            m1, m2 = st.columns(2)

            m1.metric(
                "Прогнозируемая стоимость",
                f"{prediction / 1_000_000:.1f} млн ₽"
            )

            m2.metric(
                "Прогнозируемая цена за м²",
                f"{price_per_sqm / 1000:.0f} тыс. ₽"
            )

            st.info(
                "(Прогноз является только модельной оценкой)"

            )

        except Exception as error:

            st.error(
                f"Ошибка при расчёте:\n\n"
                f"{error}"
            )
