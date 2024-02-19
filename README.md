# Federated learning framework based on Random Forest 


The main idea of the framework is to train independent Random Forests (RFs) on clients using the local data, merge independent models into a global one on the server and send it back to the clients for further use. The architecture of the proposed framework is presented bellow. The framework is evaluated for intrusion detection, more specifically for network Attacks Detection (AD) and Classification (AC) using four well-known intrusion detection datasets.  Additionally, it is evaluated how adding Differential Privacy (DP) into RF, as an additional protective mechanism, affects the framework performance. 

</br>
<img width="452" alt="image" src="https://github.com/vujicictijana/RF_FL/assets/9281983/199a21ff-642a-4548-ba72-f533c01fca3c">

</br>
</br>

Four different experiments can be found in this repository:

Data: <a href="https://www.unb.ca/cic/datasets/nsl.html">KDD</a>, <a href="https://www.unb.ca/cic/datasets/nsl.html">NSL-KDD</a>, <a href="https://research.unsw.edu.au/projects/unsw-nb15-dataset">UNSW-NB15</a>, <a href="https://www.unb.ca/cic/datasets/ids-2017.html">CIC-IDS-2017</a>

Programming language: Python

Required libraries: <a href="https://scikit-learn.org/stable/">scikit-learn</a>, <a href="https://github.com/IBM/differential-privacy-library">IBM differential privacy library</a>


Citations:

Markovic, T., Leon, M., Buffoni, D., & Punnekkat, S. (2022, June). Random forest based on federated learning for intrusion detection. In IFIP International Conference on Artificial Intelligence Applications and Innovations (pp. 132-144). Cham: Springer International Publishing.

