from flask import Flask,render_template,request,Response
import joblib
import pandas as pd
import os
BASE_DIR=os.path.dirname(os.path.abspath(__file__))

def get_path(*paths):
    return os.path.join(BASE_DIR,*paths)

app=Flask(__name__)

model=joblib.load(get_path("final_model","model.pkl"))
preprocessor=joblib.load(get_path("final_model","preprocessor.pkl"))
feature_names=joblib.load(get_path("final_model","feature_names.pkl"))

circuit_map=pd.read_csv(get_path('dataset','map.csv'))
encoder=preprocessor.named_transformers_["cat"].named_steps["encoder"]
driver_categories=encoder.categories_[0]

df_full=pd.read_csv(get_path("dataset","small_data.csv"))
df_full["Compound"] = df_full[
            ["Compound_SOFT", "Compound_MEDIUM", "Compound_HARD", "Compound_INTERMEDIATE", "Compound_WET"]
        ].idxmax(axis=1).str.replace("Compound_", "")

valid_circuits=df_full["circuitId"].unique()
filtered_map=circuit_map[circuit_map["circuitId"].isin(valid_circuits)]
circuit_dict=dict(zip(filtered_map["circuitId"],filtered_map["circuitName"]))

@app.route("/",methods=["GET","POST"])
def home():
    if request.method=="POST":
        form=request.form
        data={
           "Stint":float(form["stint"]),
            "Year":int(form["year"]),
            "Round":int(form["round"]),
            "TireAge":float(form["tire_age"]),
            "circuitId":float(form["circuit"]),
            "Driver":str(form["Driver"]),
            "grid":float(form["grid"]),
            "IsPitLap":int(form["pitlap"]),
            "HasPitOut":int(form["pitout"]),
            "LapPct":float(form["lappct"]),
            "StintLap":int(form["stintlap"]), 
        }

        tyre=form["tyre"]
        data["Compound_SOFT"]=False
        data["Compound_MEDIUM"]=False
        data["Compound_HARD"]=False
        data["Compound_INTERMEDIATE"]=False
        data["Compound_WET"]=False

        if tyre=="Soft":
            data["Compound_SOFT"]=True
        elif tyre=="Medium":
            data["Compound_MEDIUM"]=True
        elif tyre=="Hard":
            data["Compound_HARD"]=True
        elif tyre=="Intermediate":
            data["Compound_INTERMEDIATE"]=True
        elif tyre=="Wet":
            data["Compound_WET"]=True
        df=pd.DataFrame([data])
        df=df[feature_names]
        transformed=preprocessor.transform(df)
        prediction=model.predict(transformed)[0]
        
        driver=data['Driver']
        circuit=data['circuitId']
    
        tire_stats=df_full.groupby("Compound")["LapTimeSeconds"].mean()
        best_tire=tire_stats.idxmin()

        driver_df=df_full[df_full["Driver"]==driver]
        deg=driver_df.groupby("TireAge")["LapTimeSeconds"].mean().reset_index()
        deg["delta"]=deg["LapTimeSeconds"].diff()
        pit_lap=int(deg.loc[deg["delta"].idxmax(),"TireAge"])

        return render_template("index.html",lap_time=round(prediction,3),circuits=circuit_dict,drivers=driver_categories,form_data=form,best_tire=best_tire,pit_lap=pit_lap)
    return render_template("index.html",lap_time=None,circuits=circuit_dict,drivers=driver_categories,best_tire=None,pit_lap=None)

@app.route("/stats",methods=['POST'])
def stats():
    form=request.form
    data={
        "Stint":float(form["stint"]),
        "Year":int(form["year"]),
        "Round":int(form["round"]),
        "TireAge":float(form["tire_age"]),
        "circuitId":float(form["circuit"]),
        "Driver":str(form["Driver"]),
        "grid":float(form["grid"]),
        "IsPitLap":int(form["pitlap"]),
        "HasPitOut":int(form["pitout"]),
        "LapPct":float(form["lappct"]),
        "StintLap":int(form["stintlap"]), 
    }

    tyre=form["tyre"]
    data["Compound_SOFT"]=False
    data["Compound_MEDIUM"]=False
    data["Compound_HARD"]=False
    data["Compound_INTERMEDIATE"]=False
    data["Compound_WET"]=False

    if tyre=="Soft":
        data["Compound_SOFT"]=True
    elif tyre=="Medium":
        data["Compound_MEDIUM"]=True
    elif tyre=="Hard":
        data["Compound_HARD"]=True
    elif tyre=="Intermediate":
        data["Compound_INTERMEDIATE"]=True
    elif tyre=="Wet":
        data["Compound_WET"]=True
    df=pd.DataFrame([data])
    df=df[feature_names]
    transformed=preprocessor.transform(df)
    prediction=model.predict(transformed)[0]
    
    driver=data['Driver']
    circuit=data['circuitId']

    deg_df=df_full[(df_full["Driver"]==driver) &
                   (df_full['IsPitLap']==0) &
                   (df_full['HasPitOut']==0)]
    deg_data={}
    for compound in deg_df["Compound"].unique():
        temp=deg_df[deg_df["Compound"]==compound]
        grouped=temp.groupby("TireAge")["LapTimeSeconds"].mean().reset_index()
        deg_data[compound]={
            "x": grouped["TireAge"].tolist(),
            "y": grouped["LapTimeSeconds"].tolist()}
        
    driver_avg=df_full[df_full["Driver"]==driver]["LapTimeSeconds"].mean()
    circuit_avg=df_full[df_full["circuitId"]==circuit]["LapTimeSeconds"].mean()
    tire_avgs=df_full[df_full["Driver"]==driver].groupby("Compound")["LapTimeSeconds"].mean()

    return render_template("result.html",prediction=round(prediction,3),driver_avg=round(driver_avg,3),circuit_avg=round(circuit_avg,3),
                            tire_labels=list(tire_avgs.index),
                            tire_values=list(tire_avgs.values),
                            form_data=form,deg_data=deg_data)

if __name__=="__main__":
    app.run()
