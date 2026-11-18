"""Cook County Health | Chicago community health and gun-violence research dashboard.
Replace the existing repository app.py; keep the existing data/ folder.
"""
from pathlib import Path
import html
import json

import branca.colormap as bcm
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Cook County Health | Chicago Community Insights", page_icon="💜", layout="wide")
ROOT = Path(__file__).resolve().parent / "data"
PURPLE = "#592D82"
TEAL = "#087F8C"

@st.cache_data(show_spinner="Loading Cook County Health datasets...")
def load_data():
    required = ["chicago_community_area_year_merged.csv", "trauma_hospitals.csv", "dashboard_map.geojson"]
    missing = [name for name in required if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing data/ files: {', '.join(missing)}")
    df = pd.read_csv(ROOT / required[0])
    hospitals = pd.read_csv(ROOT / required[1])
    geometry = json.loads((ROOT / required[2]).read_text(encoding="utf-8"))
    df.columns = df.columns.str.strip()
    hospitals.columns = hospitals.columns.str.strip()
    essential = {"ca_num", "ca_name", "year", "shot_total", "population"}
    if not essential.issubset(df.columns):
        raise ValueError(f"Dataset missing required columns: {sorted(essential - set(df.columns))}")
    numeric = ["ca_num", "year", "shot_total", "fatal", "nonfatal", "population", "shot_rate_per_10k",
               "pct_poverty", "pct_unemployed", "per_capita_income", "pct_rent_burdened",
               "eviction_filing_rate", "profile_trauma_drive_min", "profile_clinics", "pct_on_medicaid"]
    for col in numeric:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["lat", "lon", "total_costs", "net_patient_revenue", "charity_care_cost", "uncompensated_care_cost"]:
        if col in hospitals:
            hospitals[col] = pd.to_numeric(hospitals[col], errors="coerce")
    df = df.dropna(subset=["ca_num", "ca_name", "year"]).drop_duplicates(["ca_num", "year"]).copy()
    df["ca_num"] = df["ca_num"].astype(int)
    df["year"] = df["year"].astype(int)
    if "shot_rate_per_10k" not in df:
        df["shot_rate_per_10k"] = np.where(df["population"] > 0, df["shot_total"] / df["population"] * 10000, np.nan)
    for feature in geometry.get("features", []):
        feature["properties"]["ca_num"] = int(feature["properties"]["ca_num"])
    return df, hospitals, geometry

try:
    D, H, G = load_data()
except (FileNotFoundError, ValueError, OSError, KeyError) as err:
    st.error(str(err))
    st.info("Keep the existing GitHub data/ folder beside app.py, with all three original files.")
    st.stop()

st.markdown("""<style>
.block-container {padding-top: 1.4rem; max-width: 1500px}
h1, h2, h3 {letter-spacing: -.025em}
[data-testid="stMetric"] {padding: 13px 15px; border: 1px solid rgba(110,90,130,.18); border-radius: 13px; background: var(--secondary-background-color)}
[data-testid="stSidebar"] {border-right: 3px solid #592D82}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {color: #8a52b4; border-bottom-color: #8a52b4}
.cook_county_health-banner {background: linear-gradient(110deg,#35144f,#592D82 66%,#087F8C); border-radius: 17px; color: white; padding: 24px 28px; margin-bottom: 16px}
.cook_county_health-banner h1 {color: white; margin: 0 0 6px; font-size: 2rem}
.cook_county_health-banner p {margin: 0; color: #efe5f7}
</style>""", unsafe_allow_html=True)

def nfmt(value, digits=0, prefix=""):
    try:
        return "N/A" if pd.isna(value) else prefix + f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"

def safe(row, key):
    return row.get(key, np.nan) if row is not None else np.nan

def numeric_fields(frame, names):
    return [name for name in names if name in frame.columns]

def show_chart(fig):
    fig.update_layout(margin=dict(l=12, r=12, t=20, b=12), font=dict(size=13))
    st.plotly_chart(fig, use_container_width=True)

def comparison_table(frame, first, second, metrics):
    x = frame[frame.ca_name == first].iloc[0]
    y = frame[frame.ca_name == second].iloc[0]
    rows = [{"Indicator": label, first: nfmt(safe(x, field), decimals), second: nfmt(safe(y, field), decimals)}
            for label, field, decimals in metrics if field in frame]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

YEARS = sorted(D.year.unique(), reverse=True)
year = st.sidebar.selectbox("Reporting year", YEARS)
current = D[D.year == year].copy()
areas = sorted(current.ca_name.dropna().unique().tolist())
if "selected_area" not in st.session_state or st.session_state.selected_area not in ["All Chicago"] + areas:
    st.session_state.selected_area = "All Chicago"
st.sidebar.markdown("## COOK COUNTY HEALTH")
st.sidebar.caption("Chicago community health insights")
area = st.sidebar.selectbox("Focus community area", ["All Chicago"] + areas, key="selected_area")
st.sidebar.caption(f"{len(areas)} community areas in {year} · Dataset years {min(YEARS)}–{max(YEARS)}")
st.sidebar.info("Use the map or the selector to explore a community. The selected community is shared across tabs.")

st.markdown("""<div class="cook_county_health-banner"><h1>COOK COUNTY HEALTH | Chicago Community Insights</h1>
<p>Gun violence · neighborhood conditions · healthcare access · research findings</p></div>""", unsafe_allow_html=True)
st.caption("Research prototype | Community-level descriptions and exploratory statistical associations, not causal claims")
t1, t2, t3, t4 = st.tabs(["🗺️ Gun Violence", "📊 Socioeconomic Factors", "🏥 Healthcare & Costs", "📚 Sources & Dictionary"])
selected = current if area == "All Chicago" else current[current.ca_name == area]
focused = None if area == "All Chicago" or selected.empty else selected.iloc[0]
shot = selected.shot_total.sum(min_count=1)
fatal = selected.fatal.sum(min_count=1) if "fatal" in selected else np.nan
nonfatal = selected.nonfatal.sum(min_count=1) if "nonfatal" in selected else np.nan
population = selected.population.sum(min_count=1)
rate = shot / population * 10000 if pd.notna(population) and population > 0 else np.nan

with t1:
    st.subheader(f"{area} | {year}")
    a, b, c, d = st.columns(4)
    a.metric("Shooting victims", nfmt(shot))
    b.metric("Fatal victims", nfmt(fatal))
    c.metric("Nonfatal victims", nfmt(nonfatal))
    d.metric("Victims per 10,000", nfmt(rate, 1))
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        choices = {"Shooting rate per 10,000": "shot_rate_per_10k", "Shooting victims": "shot_total",
                   "Poverty (%)": "pct_poverty", "Unemployment (%)": "pct_unemployed",
                   "Estimated trauma drive (min)": "profile_trauma_drive_min"}
        choices = {label: key for label, key in choices.items() if key in current.columns}
        label = st.selectbox("Color neighborhoods by", list(choices), key="map_metric")
        field = choices[label]
        values = current.set_index("ca_num")[field].to_dict()
        v = pd.to_numeric(current[field], errors="coerce").dropna()
        low, high = (float(v.min()), float(v.max())) if len(v) else (0., 1.)
        color = bcm.LinearColormap(["#f7f3fb", "#d2b5df", "#9a62b3", "#4d1b75"], vmin=low,
                                    vmax=max(high, low + .01), caption=label)
        mp = folium.Map(location=[41.84, -87.68], zoom_start=10, tiles="CartoDB positron", control_scale=True)
        lookup = current.set_index("ca_num")
        for feature in G.get("features", []):
            cid = feature["properties"]["ca_num"]
            if cid not in lookup.index:
                continue
            r = lookup.loc[cid]
            name = str(r.ca_name)
            value = values.get(cid, np.nan)
            active = name == area
            fill = color(value) if pd.notna(value) else "#d5d7dc"
            def styling(_feature, fill=fill, active=active):
                return {"fillColor": fill, "fillOpacity": .73, "color": PURPLE if active else "#70818d",
                        "weight": 3.5 if active else .8}
            details = [("Shooting victims", nfmt(safe(r, "shot_total"))),
                       ("Fatal victims", nfmt(safe(r, "fatal"))),
                       ("Victims per 10,000", nfmt(safe(r, "shot_rate_per_10k"), 1)),
                       ("Poverty", nfmt(safe(r, "pct_poverty"), 1) + "%"),
                       ("Unemployment", nfmt(safe(r, "pct_unemployed"), 1) + "%")]
            content = "".join(f"<tr><td>{html.escape(k)}: </td><td><b>{html.escape(val)}</b></td></tr>" for k, val in details)
            popup = folium.Popup(f"<div style='font:13px Arial;min-width:230px'><h3>{html.escape(name)}</h3><table>{content}</table></div>", max_width=320)
            folium.GeoJson(feature, style_function=styling, highlight_function=lambda f: {"weight": 3, "fillOpacity": .88},
                           tooltip=name, popup=popup).add_to(mp)
        color.add_to(mp)
        # A GeoJSON click gives the polygon tooltip via streamlit-folium; no new spatial dependency needed.
        clicked = st_folium(mp, height=540, use_container_width=True,
                            key=f"community_{year}_{label}_{area}",
                            returned_objects=["last_object_clicked_tooltip"])
        clicked_name = clicked.get("last_object_clicked_tooltip") if isinstance(clicked, dict) else None
        if isinstance(clicked_name, str) and clicked_name in areas and clicked_name != area:
            st.session_state.selected_area = clicked_name
            st.rerun()
        st.caption("Click a colored community area to focus the dashboard. Gray = missing indicator data.")
    with right:
        st.markdown("#### Communities with the most victims")
        top = current.dropna(subset=["shot_total"]).nlargest(10, "shot_total").sort_values("shot_total")
        if not top.empty:
            chart = px.bar(top, x="shot_total", y="ca_name", orientation="h", color_discrete_sequence=[TEAL],
                           labels={"shot_total": "Victims", "ca_name": ""}, height=500)
            show_chart(chart)
        st.info("**Understanding the map:** Total victims show volume; victims per 10,000 adjusts for population size.")
    if focused is not None:
        st.markdown("#### Focus community snapshot")
        cols = st.columns(4)
        for widget, (title, key) in zip(cols, [("Poverty (%)", "pct_poverty"), ("Unemployment (%)", "pct_unemployed"),
                                                ("Trauma drive estimate (min)", "profile_trauma_drive_min"),
                                                ("Clinics (estimate)", "profile_clinics")]):
            widget.metric(title, nfmt(safe(focused, key), 1))
    st.markdown("#### Shooting victims over time")
    trend = D if focused is None else D[D.ca_name == area]
    cols = numeric_fields(trend, ["fatal", "nonfatal"])
    if cols:
        trend = trend.groupby("year", as_index=False)[cols].sum()
        chart = px.bar(trend, x="year", y=cols, barmode="stack", labels={"value": "Victims", "variable": "Outcome"},
                       color_discrete_map={"fatal": "#af385a", "nonfatal": TEAL})
        chart.update_layout(xaxis=dict(dtick=1))
        show_chart(chart)
    st.caption("Shooting victims are people, not necessarily distinct shooting incidents.")

with t2:
    st.subheader("Neighborhood conditions and gun violence")
    factors = {"Poverty (%)": "pct_poverty", "Unemployment (%)": "pct_unemployed",
               "Income per person ($)": "per_capita_income", "Rent burden (%)": "pct_rent_burdened",
               "Eviction filing rate": "eviction_filing_rate"}
    factors = {label: key for label, key in factors.items() if key in current.columns}
    if factors:
        label = st.selectbox("Select a factor to compare with shooting rates", list(factors), key="factor_choice")
        key = factors[label]
        good = current.dropna(subset=[key, "shot_rate_per_10k", "shot_total"])
        if len(good) >= 2:
            chart = px.scatter(good, x=key, y="shot_rate_per_10k", size="shot_total", hover_name="ca_name",
                               color="shot_rate_per_10k", color_continuous_scale="Purples",
                               labels={key: label, "shot_rate_per_10k": "Victims per 10,000"}, height=430)
            show_chart(chart)
            r = good[[key, "shot_rate_per_10k"]].corr().iloc[0, 1]
            st.metric("Pearson correlation (r)", nfmt(r, 2))
            st.caption("Correlation shows direction and strength of a linear association. It does not establish cause and effect.")
    st.markdown("#### Compare two Chicago community areas")
    if len(areas) >= 2:
        c1, c2 = st.columns(2)
        first = c1.selectbox("Community A", areas, index=areas.index(area) if area in areas else 0)
        other_choices = [a for a in areas if a != first]
        second = c2.selectbox("Community B", other_choices, index=0)
        metrics = [("Shooting victims", "shot_total", 0), ("Victims per 10,000", "shot_rate_per_10k", 1),
                   ("Poverty (%)", "pct_poverty", 1), ("Unemployment (%)", "pct_unemployed", 1),
                   ("Income per person ($)", "per_capita_income", 0), ("Medicaid reliance (%)", "pct_on_medicaid", 1)]
        comparison_table(current, first, second, metrics)
        compare = current[current.ca_name.isin([first, second])]
        fig = px.bar(compare, x="ca_name", y="shot_rate_per_10k", color="ca_name",
                     labels={"ca_name": "Community", "shot_rate_per_10k": "Victims per 10,000"},
                     color_discrete_sequence=[PURPLE, TEAL])
        show_chart(fig)
        st.caption("Compare rates, not only totals: neighborhoods have different population sizes.")
    st.markdown("#### Multivariate regression: poverty and unemployment")
    st.write("A count regression studies whether poverty and unemployment are statistically associated with shooting-victim counts after accounting for community population and the reporting year.")
    predictors = ["pct_poverty", "pct_unemployed"]
    needed = ["year", "ca_name", "shot_total", "population"] + predictors
    if all(col in D for col in needed):
        model_data = D[needed].dropna().copy()
        model_data = model_data[(model_data.population > 0) & (model_data.shot_total >= 0)]
        if len(model_data) > 15:
            try:
                X = pd.concat([model_data[predictors].astype(float),
                               pd.get_dummies(model_data.year.astype(str), prefix="year", drop_first=True, dtype=float)], axis=1)
                X = sm.add_constant(X, has_constant="add")
                offset = np.log(model_data.population.astype(float))
                model = sm.GLM(model_data.shot_total.astype(float), X,
                               family=sm.families.Poisson(), offset=offset).fit(cov_type="HC1")
                model_data["estimated"] = model.predict(X, offset=offset)
                dispersion = model.pearson_chi2 / model.df_resid if model.df_resid > 0 else np.nan
                x1, x2, x3 = st.columns(3)
                x1.metric("Area-year observations", f"{len(model_data):,}")
                x2.metric("Poisson dispersion", nfmt(dispersion, 2))
                x3.metric("Main factors", "Poverty + unemployment")
                if pd.notna(dispersion) and dispersion > 1.5:
                    st.warning("The counts are overdispersed. The Poisson model may underestimate uncertainty. A negative-binomial model and neighborhood-level dependence checks should be considered before making statistical claims.")
                conf = model.conf_int()
                outcome = pd.DataFrame({"Factor": ["Poverty (+1 percentage point)", "Unemployment (+1 percentage point)"],
                                        "Rate ratio": np.exp(model.params[predictors].values),
                                        "95% CI lower": np.exp(conf.loc[predictors, 0].values),
                                        "95% CI upper": np.exp(conf.loc[predictors, 1].values)})
                st.dataframe(outcome.round(3), hide_index=True, use_container_width=True)
                st.caption("Rate ratio above 1 means higher fitted shooting-victim rate; below 1 means lower fitted rate. Confidence intervals are exploratory and may not address repeated observations by community.")
                observed = model_data[model_data.year == year]
                if not observed.empty:
                    fig = px.scatter(observed, x="estimated", y="shot_total", hover_name="ca_name",
                                     labels={"estimated": "Model-fitted victims", "shot_total": "Observed victims"})
                    maximum = max(observed.estimated.max(), observed.shot_total.max())
                    fig.add_shape(type="line", x0=0, y0=0, x1=maximum, y1=maximum,
                                  line=dict(color="#b0446b", dash="dash"))
                    show_chart(fig)
                st.markdown("#### What-if explorer (association, NOT a forecast)")
                chosen = area if area in areas else st.selectbox("Community to explore", areas, key="what_if_area")
                baseline = current[current.ca_name == chosen].iloc[0]
                if pd.notna(baseline.pct_poverty) and pd.notna(baseline.pct_unemployed):
                    a, b = st.columns(2)
                    poverty = a.slider("Hypothetical poverty (%)", 0., 100., float(np.clip(baseline.pct_poverty, 0, 100)), .5)
                    unemployment = b.slider("Hypothetical unemployment (%)", 0., 100., float(np.clip(baseline.pct_unemployed, 0, 100)), .5)
                    log_ratio = (model.params["pct_poverty"] * (poverty - baseline.pct_poverty) +
                                 model.params["pct_unemployed"] * (unemployment - baseline.pct_unemployed))
                    st.metric("Relative fitted rate vs. baseline", f"{np.exp(np.clip(log_ratio, -30, 30)):.2f}×")
                    st.caption("This is a mathematical model scenario. Changing unemployment or poverty in real life is NOT proven to produce this change in shootings.")
            except (ValueError, KeyError, np.linalg.LinAlgError) as error:
                st.warning(f"The exploratory model could not be fitted with this dataset: {error}")
        else:
            st.info("There are not enough complete records to fit the regression.")
    else:
        st.info("The poverty/unemployment model requires columns that are not present.")

with t3:
    st.subheader("Healthcare access and hospital financial context")
    st.warning("Travel estimates are not ambulance response times. Hospital-wide financial totals cannot be attributed specifically to shooting injuries.")
    if focused is not None:
        a, b, c = st.columns(3)
        a.metric("Nearest listed trauma center", str(safe(focused, "profile_nearest_center")))
        b.metric("Estimated trauma drive (min)", nfmt(safe(focused, "profile_trauma_drive_min"), 1))
        c.metric("Clinics (source estimate)", nfmt(safe(focused, "profile_clinics")))
    if {"lat", "lon"}.issubset(H.columns):
        hospital_map = folium.Map(location=[41.84, -87.68], zoom_start=10, tiles="CartoDB positron")
        for _, hospital in H.dropna(subset=["lat", "lon"]).iterrows():
            name = html.escape(str(safe(hospital, "hospital")))
            kind = html.escape(str(safe(hospital, "type")))
            level = html.escape(str(safe(hospital, "adult_level1_trauma")))
            folium.Marker([float(hospital.lat), float(hospital.lon)],
                          popup=folium.Popup(f"<b>{name}</b><br>Type: {kind}<br>Listed adult Level I: {level}"),
                          tooltip=str(safe(hospital, "hospital")),
                          icon=folium.Icon(color="purple", icon="plus-sign")).add_to(hospital_map)
        st_folium(hospital_map, height=390, use_container_width=True, key="hospitals_map", returned_objects=[])
        st.caption("Hospital locations and designations should be confirmed with primary sources before public use.")
    st.markdown("#### Hospital financial comparison")
    financial_cols = numeric_fields(H, ["total_costs", "net_patient_revenue", "charity_care_cost", "uncompensated_care_cost"])
    if financial_cols and "hospital" in H:
        finance_choice = st.selectbox("Choose hospital financial measure", financial_cols,
                                     format_func=lambda x: x.replace("_", " ").title())
        available = H.dropna(subset=[finance_choice, "hospital"]).sort_values(finance_choice)
        if not available.empty:
            fig = px.bar(available, x=finance_choice, y="hospital", orientation="h",
                         labels={finance_choice: "Hospital-wide USD", "hospital": ""},
                         color_discrete_sequence=[TEAL], height=max(320, len(available) * 32))
            show_chart(fig)
        st.dataframe(H[["hospital"] + financial_cols], hide_index=True, use_container_width=True)
    else:
        st.info("No comparable hospital-wide financial columns were found.")
    st.markdown("#### Healthcare neighborhood comparison")
    columns = numeric_fields(current, ["ca_name", "shot_rate_per_10k", "profile_nearest_center",
                                       "profile_trauma_drive_min", "profile_clinics", "pct_on_medicaid"])
    if columns:
        st.dataframe(current[columns].sort_values("shot_rate_per_10k", ascending=False),
                     hide_index=True, use_container_width=True)

with t4:
    st.subheader("Data sources, dictionary, and how to read the dashboard")
    st.write("The dashboard uses three local files already stored in the project repository. Exact originating organizations, extraction dates, and the time window of each non-shooting indicator should be verified against your team's source records.")
    st.dataframe(pd.DataFrame([
        {"File": "chicago_community_area_year_merged.csv", "Used for": "Victims, population, area/year observations, community profiles"},
        {"File": "dashboard_map.geojson", "Used for": "Chicago community-area boundaries"},
        {"File": "trauma_hospitals.csv", "Used for": "Listed hospitals, locations and hospital-wide finances"},
    ]), hide_index=True, use_container_width=True)
    dictionary = [
        ("ca_name", "Chicago community-area name"),
        ("ca_num", "Chicago community-area identifier"),
        ("year", "Reporting year for shooting-victim counts"),
        ("shot_total", "Number of shooting victims, not necessarily incidents"),
        ("fatal", "Victims whose shooting injuries were fatal"),
        ("nonfatal", "Victims whose shooting injuries were nonfatal"),
        ("shot_rate_per_10k", "Victims / population × 10,000"),
        ("population", "Community population denominator"),
        ("pct_poverty", "Share measured as living in poverty; source/time period requires confirmation"),
        ("pct_unemployed", "Reported unemployment percentage; definition and period require confirmation"),
        ("per_capita_income", "Income per person; source period and inflation basis require confirmation"),
        ("pct_rent_burdened", "Percentage classified as rent burdened; check source threshold"),
        ("eviction_filing_rate", "Source-reported eviction filing rate; check definition"),
        ("pct_on_medicaid", "Source-reported percentage associated with Medicaid reliance"),
        ("profile_trauma_drive_min", "Estimated driving minutes to listed trauma care, NOT emergency response time"),
        ("profile_clinics", "Source estimate of nearby clinics"),
        ("profile_nearest_center", "Nearest trauma center in source list"),
        ("total_costs", "Hospital-wide total costs, NOT gun-violence-specific costs"),
        ("charity_care_cost", "Hospital-wide charity care cost"),
        ("uncompensated_care_cost", "Hospital-wide uncompensated care cost"),
    ]
    st.dataframe(pd.DataFrame([{"Variable": a, "Plain-English meaning": b} for a, b in dictionary if a in D or a in H]),
                 hide_index=True, use_container_width=True)
    with st.expander("Plain-English regression and chart explanations", expanded=True):
        st.markdown("""
- **Correlation (r):** Describes how closely two variables move together in a linear relationship; it cannot prove one causes the other.
- **Poisson regression:** A statistical model for counts such as shooting victims. The app accounts for community population using an offset.
- **Rate ratio:** A model estimate above 1 means a higher fitted rate; below 1 means a lower fitted rate.
- **Overdispersion:** Count data vary more than a Poisson model expects. A negative-binomial alternative may fit better.
- **Observed vs. fitted:** Compares actual victim counts against the model's fitted values. Large gaps show imperfect fit.
- **Missing data:** Not every community has every variable; unavailable fields should not be read as zero.
        """)
    st.warning("Limitations: historical socioeconomic profile indicators may be repeated across years; those observations are not independent annual measurements. Cross-area correlations and this exploratory Poisson regression are not causal conclusions. Confirm source dates, cost reporting periods, and quality checks with the client.")
    st.markdown("#### Download data for review")
    st.download_button("Download selected year (CSV)", current.to_csv(index=False).encode("utf-8"),
                       file_name=f"cook_county_health_{year}_community_data.csv", mime="text/csv")
    st.download_button("Download all merged data (CSV)", D.to_csv(index=False).encode("utf-8"),
                       file_name="cook_county_health_all_years.csv", mime="text/csv")
    st.markdown("#### Dataset quick quality check")
    a, b, c = st.columns(3)
    a.metric("Years in dataset", len(YEARS))
    b.metric("Community areas this year", current.ca_num.nunique())
    c.metric("Area-year records", len(D))

st.divider()
st.caption("Cook County Health | Research and educational use. Not emergency advice, clinical guidance, hospital billing analysis, or a validated causal forecast.")
