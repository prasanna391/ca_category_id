import io
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_data():
    # Your YouTube baseline dataset
    csv_raw = """id,channelId,title,assignable
1,UCBR8-60-B28hp2BmDPdntcQ,Film & Animation,True
2,UCBR8-60-B28hp2BmDPdntcQ,Autos & Vehicles,True
10,UCBR8-60-B28hp2BmDPdntcQ,Music,True
15,UCBR8-60-B28hp2BmDPdntcQ,Pets & Animals,True
17,UCBR8-60-B28hp2BmDPdntcQ,Sports,True
18,UCBR8-60-B28hp2BmDPdntcQ,Short Movies,False
19,UCBR8-60-B28hp2BmDPdntcQ,Travel & Events,True
20,UCBR8-60-B28hp2BmDPdntcQ,Gaming,True
21,UCBR8-60-B28hp2BmDPdntcQ,Videoblogging,False
22,UCBR8-60-B28hp2BmDPdntcQ,People & Blogs,True
23,UCBR8-60-B28hp2BmDPdntcQ,Comedy,True
24,UCBR8-60-B28hp2BmDPdntcQ,Entertainment,True
25,UCBR8-60-B28hp2BmDPdntcQ,News & Politics,True
26,UCBR8-60-B28hp2BmDPdntcQ,Howto & Style,True
27,UCBR8-60-B28hp2BmDPdntcQ,Education,True
28,UCBR8-60-B28hp2BmDPdntcQ,Science & Technology,True
30,UCBR8-60-B28hp2BmDPdntcQ,Movies,False
31,UCBR8-60-B28hp2BmDPdntcQ,Anime/Animation,False
32,UCBR8-60-B28hp2BmDPdntcQ,Action/Adventure,False
33,UCBR8-60-B28hp2BmDPdntcQ,Classics,False
34,UCBR8-60-B28hp2BmDPdntcQ,Comedy,False
35,UCBR8-60-B28hp2BmDPdntcQ,Documentary,False
36,UCBR8-60-B28hp2BmDPdntcQ,Drama,False
37,UCBR8-60-B28hp2BmDPdntcQ,Family,False
38,UCBR8-60-B28hp2BmDPdntcQ,Foreign,False
39,UCBR8-60-B28hp2BmDPdntcQ,Horror,False
40,UCBR8-60-B28hp2BmDPdntcQ,Sci-Fi/Fantasy,False
41,UCBR8-60-B28hp2BmDPdntcQ,Thriller,False
42,UCBR8-60-B28hp2BmDPdntcQ,Shorts,False
43,UCBR8-60-B28hp2BmDPdntcQ,Shows,False
44,UCBR8-60-B28hp2BmDPdntcQ,Trailers,False"""

    df_base = pd.read_csv(io.StringIO(csv_raw))
    df_base['assignable'] = df_base['assignable'].astype(str)
    
    # Simulate drift: Make 'True' appear 85% of the time in production
    np.random.seed(42)
    prod_distribution = np.random.choice(['True', 'False'], size=200, p=[0.85, 0.15])
    
    records = []
    current_time = datetime.now()
    
    for idx, status in enumerate(prod_distribution):
        matched = df_base[df_base['assignable'] == status]
        # FIX: Added [0] so that we turn the selected row Series directly into a dictionary
        row = matched.sample(1).iloc[0].to_dict()
        
        # Inject performance anomalies & metadata
        row['timestamp'] = (current_time - timedelta(minutes=200 - idx)).strftime('%Y-%m-%d %H:%M:%S')
        row['api_latency_ms'] = random.uniform(80.0, 120.0) if idx % 25 == 0 else random.uniform(5.0, 15.0)
        row['http_status'] = 500 if row['api_latency_ms'] > 50.0 else 200
        row['prediction'] = status if random.random() > 0.10 else ('True' if status == 'False' else 'False')
        records.append(row)
        
    df_prod = pd.DataFrame(records)
    df_base.to_csv("baseline_dataset.csv", index=False)
    df_prod.to_csv("production_logs.csv", index=False)
    print("Created baseline_dataset.csv and drifted production_logs.csv")

if __name__ == "__main__":
    generate_data()
