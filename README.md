# PrognosAI – AI-Driven Predictive Maintenance System Using Time-Series Sensor Data

**Cohort:** Infosys Springboard 6.0  
**Contributor:** Rakesh Pedapudi  

---

## 🧩 Milestone 1: Data Preparation & Exploration

### **Overview**
This milestone focuses on collecting, exploring, and preprocessing the sensor data required for predictive maintenance modeling.

### **Files Included**
- `milestone.py` – Script to load, explore, and clean the sensor data  
- `fd1_train_preprocessed.csv` – Raw/preprocessed training data  
- `fd1_test_preprocessed.csv` – Raw/preprocessed testing data  
- `fd1_train_cleaned.csv` – Cleaned training data (output)  
- `fd1_test_cleaned.csv` – Cleaned testing data (output)

### **Data Columns**
| Column | Description |
|---------|-------------|
| `timestamp` | Date and time of measurement |
| `sensor_1`, `sensor_2`, `sensor_3` | Sensor readings (numeric) |
| `target` | Equipment health status (`0`: healthy, `1`: failure/maintenance) |

### **Steps Performed**
- Loaded CSV sensor datasets using **pandas**  
- Explored data for structure, columns, and missing values  
- Cleaned the data by removing rows with null/missing values  
- Saved cleaned datasets for use in future milestones

### **How to Run**
```bash
python milestone.py
