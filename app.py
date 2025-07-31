import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

st.set_page_config(page_title="PartnerPulse & AM Co-Pilot", layout="wide")

# ------------------- Constants -------------------
REQUIRED_COLS = [
    'Date', 'WK Number', 'OSP_Name', 'Agent_NAME',
    'Agent_AHT', 'Agent_CSAT', 'Ticket_Contribution', 'Node_Label',
    'Agent_AON', 'Agent_Part_Time_Full_Time', 'PEAK_Time', 'Agent_Tickets_Handled'
]
NUMERIC_COLS = ['Agent_AHT','Agent_CSAT','Ticket_Contribution','Agent_Tickets_Handled']
CSAT_BENCHMARK = 60
AHT_BENCHMARK = 1000

# ------------------- Helper Functions -------------------
def validate_csv(df):
    return [col for col in REQUIRED_COLS if col not in df.columns]

def clean_numeric_columns(df):
    for col in NUMERIC_COLS:
        df[col] = df[col].astype(str).str.replace('%','', regex=True).str.replace(',','', regex=True).str.strip()
        if col == 'Agent_CSAT':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def calculate_scores(df):
    node_group = df.groupby(['OSP_Name','Node_Label']).agg(
        avg_csat=('Agent_CSAT','mean'),
        avg_aht=('Agent_AHT','mean'),
        tickets_osp_node=('Agent_Tickets_Handled','sum'),
        ticket_contribution=('Ticket_Contribution','mean')
    ).reset_index()

    node_group['Weighted_CSAT'] = node_group['avg_csat'] * (node_group['ticket_contribution']/100)
    node_group['Weighted_AHT']  = node_group['avg_aht']  * (node_group['ticket_contribution']/100)

    total_tickets_by_osp = df.groupby('OSP_Name')['Agent_Tickets_Handled'].sum().reset_index(name='Total_Tickets_OSP')
    total_tickets_all = df['Agent_Tickets_Handled'].sum()

    osp_scores = node_group.groupby('OSP_Name').agg(
        Norm_CSAT=('Weighted_CSAT','sum'),
        Norm_AHT=('Weighted_AHT','sum'),
        CSAT=('avg_csat','mean'),
        AHT=('avg_aht','mean')
    ).reset_index()

    osp_scores = osp_scores.merge(total_tickets_by_osp, on='OSP_Name', how='left')
    osp_scores['Total_Tickets_Contribution'] = osp_scores['Total_Tickets_OSP'] / total_tickets_all * 100

    osp_scores['Composite_Score'] = (1 - (osp_scores['Norm_CSAT']/100)) + (osp_scores['Norm_AHT']/1000)
    osp_scores = osp_scores.sort_values(by=['Composite_Score','Total_Tickets_Contribution'], ascending=[True,False])
    osp_scores['Rank'] = range(1, len(osp_scores)+1)

    return osp_scores.round(0)

def kpi_card(metric_name, current, previous, unit="", higher_is_better=True):
    if previous is None or np.isnan(previous):
        delta_text = "—"; arrow = "→"; color = "gray"
    else:
        delta = previous - current if metric_name=="Rank" else current - previous
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"

        if metric_name=="Rank":
            color = "green" if delta > 0 else "red" if delta < 0 else "orange"
        else:
            color = "green" if (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better) else "red" if delta!=0 else "orange"
        delta_text = f"{arrow} {abs(delta):.0f}{unit}"

    return f"""
    <div style="background-color:white; border-radius:10px; padding:20px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align:center;">
        <h3 style="margin-bottom:8px;">{metric_name}</h3>
        <p style="font-size:28px; margin:0; font-weight:bold;">{current:.0f}{unit}</p>
        <p style="font-size:16px; margin:0; color:{color};">{delta_text}</p>
    </div>
    """

def color_delta(val, metric="CSAT"):
    if pd.isna(val): return ''
    if metric=="AHT":
        return 'background-color: #ff9999; text-align:center;' if val>0 else 'background-color: #b6e2b6; text-align:center;'
    else:
        return 'background-color: #b6e2b6; text-align:center;' if val>0 else 'background-color: #ff9999; text-align:center;'

def style_node(df_in):
    df_rounded = df_in.round(0)
    styled = df_rounded.style.set_properties(**{'text-align': 'center'})\
        .applymap(lambda x: color_delta(x,"CSAT"), subset=['Change_CSAT'])\
        .applymap(lambda x: color_delta(x,"AHT"), subset=['Change_AHT'])\
        .applymap(lambda x: color_delta(x,"CSAT"), subset=['Change_%Tickets'])
    return styled

def generate_weekly_strikes(df):
    weekly_csat = (
        df.groupby(['Agent_NAME','WK Number'], as_index=False)
          .agg(weekly_csat=('Agent_CSAT','mean'))
    )
    weekly_csat['Strike_Week'] = (weekly_csat['weekly_csat'] < CSAT_BENCHMARK) & (~weekly_csat['weekly_csat'].isna())
    weekly_csat = weekly_csat.sort_values(['Agent_NAME','WK Number'])
    weekly_csat['Consec_Strikes'] = (
        weekly_csat.groupby('Agent_NAME')['Strike_Week']
        .apply(lambda x: x.astype(int).groupby((x != x.shift()).cumsum()).cumsum())
        .reset_index(drop=True)
    )
    return weekly_csat

def generate_action_items(df, osp_scores):
    actions = {}
    weekly_csat = generate_weekly_strikes(df)
    for _, row in osp_scores.iterrows():
        osp = row['OSP_Name']
        osp_data = df[df['OSP_Name'] == osp]
        osp_actions, agent_actions = [], []

        if row['CSAT'] < CSAT_BENCHMARK:
            diff = CSAT_BENCHMARK - row['CSAT']
            osp_actions.append(f"OSP {osp} underperforming in CSAT → Improve by {diff:.0f}% to meet benchmark.")
        if row['AHT'] > AHT_BENCHMARK:
            diff = row['AHT'] - AHT_BENCHMARK
            osp_actions.append(f"OSP {osp} underperforming in AHT → Reduce by {diff:.0f}s to meet benchmark.")

        agents_in_osp = osp_data['Agent_NAME'].unique()
        for agent in agents_in_osp:
            agent_weeks = weekly_csat[weekly_csat['Agent_NAME']==agent]
            max_strike = agent_weeks['Consec_Strikes'].max()
            if max_strike >= 3:
                agent_actions.append(f"Agent {agent} 3 strikes → Penalise")
            elif max_strike >= 1:
                agent_actions.append(f"Agent {agent} underperforming → Coaching needed")

        actions[osp] = {"OSP": osp_actions, "Agent": agent_actions}

    return actions

def generate_account_manager_actions(df):
    actions = {}
    node_group = df.groupby(['OSP_Name','Node_Label']).agg(
        avg_csat=('Agent_CSAT','mean'),
        avg_aht=('Agent_AHT','mean')
    ).reset_index()

    for osp in df['OSP_Name'].unique():
        osp_actions = []
        osp_nodes = node_group[node_group['OSP_Name']==osp]
        for _, row in osp_nodes.iterrows():
            node = row['Node_Label']
            if row['avg_csat'] < CSAT_BENCHMARK:
                osp_actions.append(f"Node {node}: CSAT {row['avg_csat']:.0f}% → Follow up & reallocate tickets.")
            if row['avg_aht'] > AHT_BENCHMARK:
                osp_actions.append(f"Node {node}: AHT {row['avg_aht']:.0f}s → Follow up & reallocate tickets.")
        actions[osp] = osp_actions

    return actions

# ------------------- Data Upload -------------------
st.sidebar.title("Data Upload & Filters")
reset = st.sidebar.button("🔄 Reset Dashboard")
if reset:
    st.experimental_rerun()

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")
csv_data = st.sidebar.text_area("Or paste CSV data here")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV file uploaded successfully!")
elif csv_data.strip() != "":
    df = pd.read_csv(io.StringIO(csv_data))
    st.success("CSV data pasted successfully!")
else:
    st.warning("Please upload a CSV or paste CSV data to start.")
    st.stop()

df = clean_numeric_columns(df)
df_full = df.copy()
all_weeks = sorted(df['WK Number'].unique())

# Sidebar filters
current_week = st.sidebar.selectbox("Select Primary Week", all_weeks, index=len(all_weeks)-1)
agent_aon_sel = st.sidebar.multiselect("Filter Agent AON", df['Agent_AON'].unique(), default=list(df['Agent_AON'].unique()))
agent_pt_sel = st.sidebar.multiselect("Filter Agent Type", df['Agent_Part_Time_Full_Time'].unique(), default=list(df['Agent_Part_Time_Full_Time'].unique()))
peak_sel = st.sidebar.multiselect("Filter Peak Time", df['PEAK_Time'].unique(), default=list(df['PEAK_Time'].unique()))
osp_sel = st.sidebar.multiselect("Filter OSP Name", df['OSP_Name'].unique(), default=[])

df = df_full[
    (df_full['Agent_AON'].isin(agent_aon_sel)) &
    (df_full['Agent_Part_Time_Full_Time'].isin(agent_pt_sel)) &
    (df_full['PEAK_Time'].isin(peak_sel)) &
    (df_full['WK Number'] == current_week)
]
if osp_sel:
    df = df[df['OSP_Name'].isin(osp_sel)]

if df.empty:
    st.error("No data available for the selected filters.")
    st.stop()

osp_scores = calculate_scores(df)
action_items = generate_action_items(df, osp_scores)
weekly_csat = generate_weekly_strikes(df_full)

# ------------------- Tabs -------------------
tab1, tab2 = st.tabs(["OSP View", "Account Manager View"])

with tab1:
    st.header("OSP Performance View")
    selected_osp = st.selectbox("Select OSP", osp_scores['OSP_Name'])
    osp_data = df[df['OSP_Name']==selected_osp]

    comparison_week = st.selectbox("Compare With Week (Optional)", [None]+list(all_weeks))
    comparison_df = df_full[df_full['WK Number']==comparison_week] if comparison_week else pd.DataFrame()
    prev_scores = calculate_scores(comparison_df) if not comparison_df.empty else None

    # KPI Cards
    row = osp_scores[osp_scores['OSP_Name']==selected_osp].iloc[0]
    prev_row = None
    if prev_scores is not None and selected_osp in prev_scores['OSP_Name'].values:
        prev_row = prev_scores[prev_scores['OSP_Name']==selected_osp].iloc[0]

    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(kpi_card("Rank", row['Rank'], prev_row['Rank'] if prev_row is not None else np.nan, unit="", higher_is_better=False), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("AHT (s)", row['AHT'], prev_row['AHT'] if prev_row is not None else np.nan, unit="s", higher_is_better=False), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("CSAT (%)", row['CSAT'], prev_row['CSAT'] if prev_row is not None else np.nan, unit="%", higher_is_better=True), unsafe_allow_html=True)

    # Node-level metrics table
    total_current = osp_data['Agent_Tickets_Handled'].sum()
    current_nodes = osp_data.groupby(['WK Number','Node_Label']).agg(
        Tickets=('Agent_Tickets_Handled','sum'),
        CSAT=('Agent_CSAT','mean'),
        AHT=('Agent_AHT','mean')
    ).reset_index()
    current_nodes['%Tickets'] = (current_nodes['Tickets']/total_current*100) if total_current>0 else 0

    if not comparison_df.empty:
        prev_data = comparison_df[comparison_df['OSP_Name']==selected_osp]
        total_prev = prev_data['Agent_Tickets_Handled'].sum()
        prev_nodes = prev_data.groupby(['Node_Label']).agg(
            Tickets=('Agent_Tickets_Handled','sum'),
            CSAT=('Agent_CSAT','mean'),
            AHT=('Agent_AHT','mean')
        ).reset_index()
        prev_nodes['%Tickets'] = (prev_nodes['Tickets']/total_prev*100) if total_prev>0 else 0
        merged = current_nodes.merge(prev_nodes, on='Node_Label', how='left', suffixes=('','_Prev'))
        merged['Change_CSAT'] = merged['CSAT']-merged['CSAT_Prev']
        merged['Change_AHT'] = merged['AHT']-merged['AHT_Prev']
        merged['Change_%Tickets'] = merged['%Tickets']-merged['%Tickets_Prev']
    else:
        merged = current_nodes.copy()
        merged[['Change_CSAT','Change_AHT','Change_%Tickets']] = np.nan

    final_table = merged[['WK Number','Node_Label','%Tickets','CSAT','AHT','Change_CSAT','Change_AHT','Change_%Tickets']]
    st.dataframe(style_node(final_table), use_container_width=True)

    with st.expander("🏆 Top 25 & Bottom 25 Agents"):
        strike_summary = weekly_csat.groupby('Agent_NAME')['Consec_Strikes'].max().reset_index(name='Strikes')
        osp_data_strike = osp_data.merge(strike_summary, on='Agent_NAME', how='left')
        top_agents = osp_data_strike.sort_values('Agent_CSAT',ascending=False).head(25).round(0)
        bottom_agents = osp_data_strike.sort_values('Agent_CSAT',ascending=True).head(25).round(0)
        st.write("**Top 25 Agents**"); st.dataframe(top_agents[['Agent_NAME','Agent_CSAT','Agent_AHT','Strikes']].style.set_properties(**{'text-align':'center'}))
        st.write("**Bottom 25 Agents**"); st.dataframe(bottom_agents[['Agent_NAME','Agent_CSAT','Agent_AHT','Strikes']].style.set_properties(**{'text-align':'center'}))

    with st.expander("💡 Action Items"):
        st.write("### 🏢 OSP Level Actions")
        for item in action_items[selected_osp]["OSP"]: st.write("- ", item)
        st.write("### 👤 Agent Level Actions")
        for item in action_items[selected_osp]["Agent"]: st.write("- ", item)

with tab2:
    st.header("Account Manager View")
    st.dataframe(osp_scores.style.set_properties(**{'text-align': 'center'}))

    with st.expander("📊 CSAT and AHT by OSP & PEAK_Time"):
        csat_pivot = df.pivot_table(index='OSP_Name', columns='PEAK_Time', values='Agent_CSAT', aggfunc='mean').round(0)
        aht_pivot = df.pivot_table(index='OSP_Name', columns='PEAK_Time', values='Agent_AHT', aggfunc='mean').round(0)

        csat_pivot['Total'] = csat_pivot.mean(axis=1)
        aht_pivot['Total'] = aht_pivot.mean(axis=1)
        csat_pivot.loc['Total'] = csat_pivot.mean(axis=0)
        aht_pivot.loc['Total'] = aht_pivot.mean(axis=0)

        def csat_color(val):
            if pd.isna(val): return ''
            light_green = '#b6e2b6'; light_orange = '#ffb3b3'; light_red = '#F08080'
            if val >= 75: color = light_green
            elif val < 70: color = light_red
            else: color = light_orange
            return f'background-color: {color}; color: black; text-align: center;'

        def aht_color(val):
            if pd.isna(val): return ''
            light_green = '#b6e2b6'; light_orange = '#ffb3b3'; light_red = '#F08080'
            if val <= AHT_BENCHMARK: color = light_green
            elif val > AHT_BENCHMARK*1.1: color = light_red
            else: color = light_orange
            return f'background-color: {color}; color: black; text-align: center;'

        csat_styled = csat_pivot.style.applymap(csat_color).format("{:.0f}%").set_table_styles(
            [{"selector": "th", "props": [("background-color", "#007BFF"), ("color", "black")]}]
        )
        aht_styled = aht_pivot.style.applymap(aht_color).format("{:.0f}").set_table_styles(
            [{"selector": "th", "props": [("background-color", "#007BFF"), ("color", "black")]}]
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("CSAT by OSP and PEAK_Time")
            st.dataframe(csat_styled, use_container_width=True)
        with col2:
            st.subheader("AHT by OSP and PEAK_Time")
            st.dataframe(aht_styled, use_container_width=True)

    with st.expander("📈 Weekly CSAT and AHT Trends (All OSPs)"):
        col1, col2 = st.columns(2)
        csat_weekly = df_full.groupby('WK Number', as_index=False)['Agent_CSAT'].mean()
        with col1:
            st.plotly_chart(px.line(csat_weekly, x='WK Number', y='Agent_CSAT', title='Weekly CSAT Trend (All OSP)', markers=True), use_container_width=True)
        aht_weekly = df_full.groupby('WK Number', as_index=False)['Agent_AHT'].mean()
        with col2:
            st.plotly_chart(px.line(aht_weekly, x='WK Number', y='Agent_AHT', title='Weekly AHT Trend (All OSP)', markers=True), use_container_width=True)

    with st.expander("💡 Account Manager Action Items by OSP"):
        am_actions_grouped = generate_account_manager_actions(df)
        for osp, actions in am_actions_grouped.items():
            st.write(f"### {osp}")
            for act in actions: st.write("- ", act)

# ---------------- Download Buttons ----------------
st.download_button("Download Processed OSP Scores", osp_scores.round(0).to_csv(index=False), "osp_scores.csv", "text/csv")
st.download_button("Download Annotated Agent Data", df.round(0).to_csv(index=False), "agent_data.csv", "text/csv")
