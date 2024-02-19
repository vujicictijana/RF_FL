# Federated learning framework based on Random Forest 


The main idea of the framework is to train independent Random Forests (RFs) on clients using the local data, merge independent models into a global one on the server and send it back to the clients for further use. The architecture of the proposed framework is presented bellow. The framework is evaluated for intrusion detection, more specifically for network Attacks Detection (AD) and Classification (AC) using four well-known intrusion detection datasets.  Additionally, it is evaluated how adding Differential Privacy (DP) into RF, as an additional protective mechanism, affects the framework performance. 

</br>
<img width="452" alt="image" src="https://github.com/vujicictijana/RF_FL/assets/9281983/199a21ff-642a-4548-ba72-f533c01fca3c">

</br>
</br>

<b>Four different experiments can be found in this repository:</b>
<ul>
  <li> <b>Experiment 1 - Selection of RF hyper-parameters:</b> The experiment was conducted before splitting the datasets into subsets, with the goal of finding the best combination of RF hyper-parameters for a specific dataset and specific problem. Hyper-parameters that were tested include the number of Decision Trees (DTs) (odd numbers between 1 and 100), splitting rule (gini or entropy), and ensemble method (SV or WV). The best combination of hyper-parameters that was discovered in this experiment was used as the RF setup for all subsequent experiments for the specific problem on a specific dataset. File: <i> Experiment1.ipynb </i></li>
  <li> <b>Experiment 2 -  Evaluation of independent RFs on different clients: </b> For each client an independent RF was trained on data from its subset, using the best combination of hyper-parameters from Experiment 1. Different methods of obtaining subsets were tested in this experiment:
    <ul>
         <li> Experiment 2.1 - Subsets obtained using a specific feature as a division criteria.   </li>
          <li> Experiment 2.2 - Subsets obtained using random division of data among clients, such that each client gets the same amount of data.  </li>
          <li> Experiment 2.3 - Subsets obtained using random division of data among clients, such that each client gets the same amount of data as in the Experiment 2.1.</li>
    </ul>
    File: <i> Experiment2.ipynb </i>
  <li> <b>Experiment 3 -  Global RF based on Federated Learning: </b> Independent RFs were combined into a global one using four different merging methods RF\_S\_DTs\_A, RF\_S\_DTs\_WA, RF\_S\_DTs\_A\_All, RF\_S\_DTs\_WA\_All (check the <a href="https://drive.google.com/file/d/1E0BgUdOfqnj9UOrbwW4kRcRex4EFntNa/view"> reference </a> for detailed explanation) and varying number of DTs. The global RF was tested on the entire testing set and the performances of global RF were compared with the performances of independent RFs on the entire testing set. Files: <i> Experiment3.ipynb, Experiment3ResultsCSV</i></li>
  <li>  <b>Experiment 4 -  Global RF with differential privacy based on Federated Learning: </b> Independent RF with DP was trained for each client on data from its subset (with respect to the division criteria) and tested on the entire testing set. Four different values of  &epsilon; parameter were tested: 0.1, 0.5, 1 and 5. After that, the 
    independent RFs were combined into a global one using the combination of the merging method and the number of DTs that had the best performance in Experiment 3 for the specific problem in the specific data set.     The global RF was tested on the entire testing set and the results and performances of global RF were compared with the performances of independent RFs with differential privacy.  Files: <i> Experiment2DP.ipynb, Experiment3DP.ipynb  </i>
</ul>

Data: <a href="https://www.unb.ca/cic/datasets/nsl.html">KDD</a>, <a href="https://www.unb.ca/cic/datasets/nsl.html">NSL-KDD</a>, <a href="https://research.unsw.edu.au/projects/unsw-nb15-dataset">UNSW-NB15</a>, <a href="https://www.unb.ca/cic/datasets/ids-2017.html">CIC-IDS-2017</a>

Programming language: Python

Required libraries: <a href="https://scikit-learn.org/stable/">scikit-learn</a>, <a href="https://github.com/IBM/differential-privacy-library">IBM differential privacy library</a>


Citations:

Markovic, T., Leon, M., Buffoni, D., & Punnekkat, S. (2022, June). <a href="https://drive.google.com/file/d/1E0BgUdOfqnj9UOrbwW4kRcRex4EFntNa/view">Random forest based on federated learning for intrusion detection.</a> In IFIP International Conference on Artificial Intelligence Applications and Innovations (pp. 132-144). Cham: Springer International Publishing.

