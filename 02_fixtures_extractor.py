# @title Extração jogos futuros
import re
from datetime import datetime
import pandas as pd
from curl_cffi import requests

def get_fixtures_dataframe(fixtures_url: str) -> pd.DataFrame:
    """
    Baixa os jogos futuros usando curl_cffi e faz o parse de rodada, times,
    data/horário e URLs dos escudos de mandante e visitante.
    """
    r = requests.get(fixtures_url, impersonate="chrome120")
    if r.status_code != 200:
        raise Exception(f"Erro ao acessar {fixtures_url}: status {r.status_code}")

    html = r.text
    # Pega o maior bloco de feed contendo os jogos futuros
    feeds = re.findall(r'SA÷1¬~ZA÷.*', html)
    if not feeds:
        raise Exception("Nenhum feed de jogos encontrado no HTML retornado.")

    # O bloco com maior quantidade de dados contém toda a lista de jogos futuros
    main_feed = max(feeds, key=len)
    
    matches_raw = main_feed.split("~AA÷")[1:]
    records = []

    for m in matches_raw:
        def get_field(code: str) -> str:
            match = re.search(rf"(?:^|¬){code}÷([^¬]*)", m)
            return match.group(1) if match else ""

        match_id = m.split("¬")[0]
        timestamp_str = get_field("AD")
        round_name = get_field("ER")
        home_team = get_field("AE")
        away_team = get_field("AF")
        home_logo_id = get_field("OA")
        away_logo_id = get_field("OB")

        match_date = None
        if timestamp_str and timestamp_str.isdigit():
            match_date = datetime.fromtimestamp(int(timestamp_str))

        round_num = None
        if round_name:
            match_rnd = re.search(r"\d+", round_name)
            if match_rnd:
                round_num = int(match_rnd.group())

        home_logo_url = f"https://static.flashscore.com/res/image/data/{home_logo_id}" if home_logo_id else None
        away_logo_url = f"https://static.flashscore.com/res/image/data/{away_logo_id}" if away_logo_id else None

        records.append({
            "Match_ID": match_id,
            "Round": round_name,
            "Round_Num": round_num,
            "Date": match_date.strftime("%Y-%m-%d %H:%M") if match_date else None,
            "Home": home_team,
            "Away": away_team,
            "Home_Logo": home_logo_url,
            "Away_Logo": away_logo_url,
        })

    df_futuro = pd.DataFrame(records)
    # Remove duplicidades e ordena por Rodada e Data
    df_futuro = df_futuro.drop_duplicates(subset=["Match_ID"]).sort_values(by=["Round_Num", "Date"]).reset_index(drop=True)

    # Jogos adiados da Rodada 21 sem data confirmada que o Flashscore omite do feed de fixtures
    postponed_matches_r21 = [
        {"Match_ID": "POSTP_SPFC_SAN", "Round": "Round 21", "Round_Num": 21, "Date": None, "Home": "Sao Paulo", "Away": "Santos", "Home_Logo": None, "Away_Logo": None},
        {"Match_ID": "POSTP_CAM_RBB", "Round": "Round 21", "Round_Num": 21, "Date": None, "Home": "Atletico-MG", "Away": "Bragantino", "Home_Logo": None, "Away_Logo": None},
        {"Match_ID": "POSTP_CHA_VAS", "Round": "Round 21", "Round_Num": 21, "Date": None, "Home": "Chapecoense-SC", "Away": "Vasco", "Home_Logo": None, "Away_Logo": None},
    ]

    for pm in postponed_matches_r21:
        # Verifica se o confronto já está presente
        exists = ((df_futuro["Home"] == pm["Home"]) & (df_futuro["Away"] == pm["Away"])).any()
        if not exists:
            df_futuro = pd.concat([pd.DataFrame([pm]), df_futuro], ignore_index=True)

    return df_futuro.sort_values(by=["Round_Num", "Date"]).reset_index(drop=True)

if __name__ == "__main__":
    url = "https://www.flashscore.com/football/brazil/serie-a-betano/fixtures/"
    df_futuro = get_fixtures_dataframe(url)
    print(f"Total de jogos futuros extraídos: {len(df_futuro)}")
    print(f"Rodadas encontradas: {sorted(df_futuro['Round_Num'].dropna().unique())}")
    print("\nPreview dos primeiros jogos:")
    print(df_futuro[["Round", "Date", "Home", "Away", "Home_Logo"]].head(10))
