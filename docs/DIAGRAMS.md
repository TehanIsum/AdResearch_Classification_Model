# System Workflow Diagrams (ASCII Art)

Visual representations of system workflows and processes.

## 1. Overall System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                               │
│                                                                       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐    │
│  │  CLI Mode      │  │  Interactive   │  │  Prediction Mode   │    │
│  │  (CSV Input)   │  │  Menu          │  │  (Single Ad)       │    │
│  └────────┬───────┘  └────────┬───────┘  └──────────┬─────────┘    │
└───────────┼──────────────────┼─────────────────────┼───────────────┘
            │                  │                     │
            └──────────────────┴─────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   MAIN APPLICATION    │
                    │     (main.py)         │
                    │  AdDisplaySystem      │
                    └───────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼────────┐    ┌────────▼────────┐    ┌────────▼────────┐
│  Classifier    │    │ Weather Service │    │ Recommendation  │
│   Module       │    │     Module      │    │    Engine       │
│ (classifier.py)│    │(weather_svc.py) │    │  (rec_engine.py)│
└───────┬────────┘    └────────┬────────┘    └────────┬────────┘
        │                      │                       │
┌───────▼────────┐    ┌────────▼────────┐    ┌────────▼────────┐
│  ML Model      │    │  Weather API    │    │  Ads Database   │
│  (.pkl files)  │    │ OpenWeatherMap  │    │    (CSV)        │
└────────────────┘    └─────────────────┘    └─────────────────┘
```

## 2. Ad Display Flow (Main Use Case)

```
           START
             │
             ▼
    ┌────────────────┐
    │  Load System   │
    │  - Ads DB      │
    │  - Weather Svc │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │  Read CSV Row  │
    │  (Target Vals) │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐      ┌──────────────┐
    │ Weather Needed?│──Yes─▶│  Fetch       │
    └────────┬───────┘      │  Weather API │
          No │               └──────┬───────┘
             │◀─────────────────────┘
             ▼
    ┌────────────────┐
    │ Build Target   │
    │ Profile        │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │ Find Best Ad   │
    │ (Score 0-4)    │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │  Display Ad    │
    │  (3 seconds)   │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │  More Rows?    │
    └────┬──────┬────┘
       Yes│     │No
          │     ▼
          │  ┌────────┐
          │  │  END   │
          │  └────────┘
          │
          └────┐
               │
               ▼
      (Loop back to Read CSV Row)
```

## 3. Classification Prediction Flow

```
        START: New Ad Title
              │
              ▼
     ┌────────────────┐
     │ Load ML Model  │
     │ (if not loaded)│
     └────────┬───────┘
              │
              ▼
     ┌────────────────┐
     │  Clean Text    │
     │  - Lowercase   │
     │  - Remove spcl │
     └────────┬───────┘
              │
              ▼
     ┌────────────────┐
     │   TF-IDF       │
     │ Vectorization  │
     │ (5000 features)│
     └────────┬───────┘
              │
              ▼
     ┌────────────────┐
     │ Random Forest  │
     │ Multi-Output   │
     │ Classifier     │
     └────────┬───────┘
              │
              ▼
     ┌────────────────┐
     │ Decode Labels  │
     │ - Age Group    │
     │ - Gender       │
     │ - Mood         │
     │ - Weather      │
     └────────┬───────┘
              │
              ▼
     ┌────────────────┐
     │ Display Results│
     └────────────────┘
              │
              ▼
            END
```

## 4. Weather Integration Flow

```
         Need Weather?
              │
              ▼
     ┌────────────────┐
     │ Check API Key  │
     └────┬──────┬────┘
       Yes│     │No
          │     ▼
          │  ┌────────────┐
          │  │Use Default │
          │  │"sunny"     │
          │  └────────────┘
          │        │
          ▼        │
  ┌───────────┐   │
  │ API Call  │   │
  │ (HTTP GET)│   │
  └─────┬─────┘   │
        │         │
        ▼         │
  ┌───────────┐  │
  │Parse JSON │  │
  │Response   │  │
  └─────┬─────┘  │
        │         │
        ▼         │
  ┌───────────┐  │
  │Categorize │  │
  │Weather    │  │
  │           │  │
  │Rain? →    │  │
  │  rainy    │  │
  │Cold/Snow? │  │
  │  cold     │  │
  │Else →     │  │
  │  sunny    │  │
  └─────┬─────┘  │
        │         │
        └────┬────┘
             │
             ▼
    ┌────────────────┐
    │Return Category │
    └────────────────┘
```

## 5. Ad Matching Algorithm

```
    Input: Target Demographics
              │
              ▼
    ┌─────────────────┐
    │ Initialize      │
    │ best_score = 0  │
    │ best_ad = None  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ FOR EACH Ad     │◄────────┐
    │ in Database     │         │
    └────────┬────────┘         │
             │                  │
             ▼                  │
    ┌─────────────────┐        │
    │ Calculate Score │        │
    │                 │        │
    │ score = 0       │        │
    │                 │        │
    │ Age Match?      │        │
    │   score += 1    │        │
    │ Gender Match?   │        │
    │   score += 1    │        │
    │ Mood Match?     │        │
    │   score += 1    │        │
    │ Weather Match?  │        │
    │   score += 1    │        │
    └────────┬────────┘        │
             │                  │
             ▼                  │
    ┌─────────────────┐        │
    │ score >         │        │
    │ best_score?     │        │
    └────┬──────┬─────┘        │
       No│     │Yes            │
         │     ▼               │
         │ ┌─────────────┐    │
         │ │Update best  │    │
         │ │best_score   │    │
         │ │best_ad      │    │
         │ └─────────────┘    │
         │                     │
         ▼                     │
    ┌─────────────────┐       │
    │ More Ads?       │       │
    └────┬──────┬─────┘       │
       Yes│     │No            │
          └─────┘              │
                │
                ▼
    ┌─────────────────┐
    │ Return best_ad  │
    │ with score      │
    └─────────────────┘
```

## 6. Model Training Pipeline (Google Colab)

```
         Google Colab
              │
              ▼
    ┌─────────────────┐
    │ Upload Dataset  │
    │     (CSV)       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Preprocess      │
    │ - Load CSV      │
    │ - Clean Text    │
    │ - Remove Dups   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Feature Eng     │
    │ - TF-IDF        │
    │ - Label Encode  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Train-Test      │
    │ Split (80/20)   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Train Model     │
    │ Random Forest   │
    │ (5-10 minutes)  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Evaluate        │
    │ - Accuracy      │
    │ - Per Category  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Save Files      │
    │ - model.pkl     │
    │ - vectorizer.pkl│
    │ - encoders.pkl  │
    │ - metadata.pkl  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Download Files  │
    │ to Local PC     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Place in        │
    │ models/         │
    │ directory       │
    └─────────────────┘
```

## 7. Interactive Mode Navigation

```
                    MAIN MENU
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    1                   2                   3
    │                   │                   │
    ▼                   ▼                   ▼
┌─────────┐      ┌──────────┐      ┌──────────┐
│Display  │      │ Predict  │      │  Test    │
│Ads from │      │Categories│      │ Weather  │
│CSV      │      │for Ad    │      │ Service  │
└────┬────┘      └────┬─────┘      └────┬─────┘
     │                │                  │
     │                │                  │
     ▼                ▼                  ▼
┌─────────┐      ┌──────────┐      ┌──────────┐
│Process  │      │Load Model│      │Fetch &   │
│CSV File │      │Predict   │      │Display   │
│Display  │      │Show      │      │Weather   │
│Ads      │      │Results   │      │Info      │
└────┬────┘      └────┬─────┘      └────┬─────┘
     │                │                  │
     └────────────────┴──────────────────┘
                      │
                      ▼
               Back to Main Menu
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    4                 │                 5
    │                 │                 │
    ▼                 │                 ▼
┌─────────┐          │           ┌──────────┐
│ View    │          │           │   Exit   │
│Database │          │           │  System  │
│Stats    │          │           └──────────┘
└────┬────┘          │
     │               │
     ▼               │
┌─────────┐          │
│Show Ad  │          │
│Counts & │          │
│Distrib. │          │
└────┬────┘          │
     │               │
     └───────────────┘
           │
           ▼
    Back to Main Menu
```

## 8. Data Flow Diagram

```
    ┌─────────────┐
    │ User/System │
    └──────┬──────┘
           │
           │ 1. Input
           ▼
    ┌─────────────────┐
    │  Target Values  │
    │   (CSV/CLI)     │
    └────────┬────────┘
             │
             │ 2. Read
             ▼
    ┌─────────────────┐        ┌──────────────┐
    │ Main Application│───────▶│ Weather API  │
    │  (Controller)   │  3a    │  (External)  │
    └────┬──────┬─────┘        └──────┬───────┘
         │      │                     │
    3b   │      │ 3c                  │ 3a'
         │      │              Weather│Category
         │      │                     │
         ▼      ▼                     ▼
    ┌────────┐ ┌─────────┐     ┌────────────┐
    │ML Model│ │Ads DB   │     │Weather Svc │
    │(.pkl)  │ │(CSV)    │     │(Module)    │
    └───┬────┘ └────┬────┘     └──────┬─────┘
        │           │                  │
        │ 4a        │ 4b               │ 4c
        │Prediction │Best Ad           │Category
        ▼           ▼                  │
    ┌─────────────────────────────────┴┐
    │      Recommendation Engine        │
    │    (Matching & Selection)         │
    └──────────────┬────────────────────┘
                   │
                   │ 5. Selected Ad
                   ▼
    ┌──────────────────────────────────┐
    │         Display System            │
    │   (Terminal Output + Timer)       │
    └──────────────┬───────────────────┘
                   │
                   │ 6. Output
                   ▼
            ┌──────────────┐
            │    User      │
            │  (Viewer)    │
            └──────────────┘
```

## 9. Error Handling Flow

```
        Operation Attempt
              │
              ▼
    ┌─────────────────┐
    │ Try Operation   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Success?        │
    └────┬──────┬─────┘
       Yes│     │No
          │     │
          │     ▼
          │ ┌──────────────┐
          │ │ Log Error    │
          │ └──────┬───────┘
          │        │
          │        ▼
          │ ┌──────────────┐
          │ │ Recoverable? │
          │ └───┬──────┬───┘
          │   Yes│     │No
          │     │      │
          │     ▼      ▼
          │ ┌─────┐ ┌──────┐
          │ │Use  │ │Show  │
          │ │Def  │ │Error │
          │ │ault │ │&Exit │
          │ └──┬──┘ └──────┘
          │    │
          └────┴────┐
                    │
                    ▼
           ┌────────────────┐
           │ Continue       │
           └────────────────┘
```

## 10. System State Diagram

```
    ┌───────────┐
    │   START   │
    └─────┬─────┘
          │
          ▼
    ┌───────────┐
    │Initializing│
    └─────┬─────┘
          │
          ▼
    ┌───────────┐
    │   Ready   │◄────────────┐
    └─────┬─────┘             │
          │                   │
    ┌─────┴─────┐             │
    │           │             │
    ▼           ▼             │
┌────────┐  ┌────────┐       │
│Display │  │Predict │       │
│Mode    │  │Mode    │       │
└───┬────┘  └───┬────┘       │
    │           │             │
    ▼           ▼             │
┌────────┐  ┌────────┐       │
│Process │  │Classify│       │
│ing     │  │ing     │       │
└───┬────┘  └───┬────┘       │
    │           │             │
    └─────┬─────┘             │
          │                   │
          ▼                   │
    ┌───────────┐             │
    │ Complete  │             │
    └─────┬─────┘             │
          │                   │
    ┌─────┴─────┐             │
    │           │             │
    │Continue?  │             │
    │           │             │
    └─┬───────┬─┘             │
    Yes│     │No              │
       │     │                │
       └─────┘                │
             │                │
             ▼                │
       ┌───────────┐          │
       │    END    │          │
       └───────────┘          │
                              │
    (Loop back to Ready) ─────┘
```

---

**Note**: These ASCII diagrams provide visual representations of the system workflows. For more detailed explanations, refer to the corresponding documentation files.

**Files Referenced**:
- System Architecture: `ARCHITECTURE.md`
- Detailed Workflows: `WORKFLOW.md`
- API Usage: `API_REFERENCE.md`
- User Guide: `README.md`

**Last Updated**: November 2025
