# LODAP: On-Device Incremental Learning Via Lightweight Operations and Data Pruning
LODAP is a lightweight incremental learning framework for edge devices that utilizes efficient incremental modules (EIM) and data pruning strategy to significantly improve the learning accuracy of new categories while reducing model complexity and training overhead.
## ✨ Key highlights
1.**​Efficient Incremental Module (EIM)​**​: By using lightweight adapters and structural reparameterization techniques, incremental learning of new categories can be achieved with low computational overhead.\
2.​**​Data Pruning​**: Selecting high-value training samples based on EL2N score significantly reduces training costs while maintaining model accuracy.
## 🚀 Quick start
1. Clone the repository:
```bash
git clone https://github.com/your-repo/LODAP.git
cd LODAP
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the experiment:
```bash
python main.py
```
## 🔧 Key parameters

| Parameter          | Description                           | Default         |
|--------------------|---------------------------------------|-----------------|
| `--data_name`      | Dataset name to use                   | cifar100      |
| `--total_nc`       | Class number for the dataset          | 100             |
| `--fg_nc`          | The number of classes in first task   | 50              |
| `--task_num`       | The number of incremental steps       | 10              |
| `--prune_fraction` | The ratio of pruned weights           | 0.7             |





