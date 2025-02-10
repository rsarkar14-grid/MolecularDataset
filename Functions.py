import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Fragments
from rdkit.Chem import Lipinski
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import DataStructs
from multiprocessing import Pool, cpu_count
from chembl_webresource_client.new_client import new_client
from functools import partial
from IFG import identify_functional_groups
from sklearn.manifold import MDS
from io import StringIO
from tqdm import tqdm
from collections import Counter
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
import pickle
import io
from PIL import Image
import mysqlx

# Compute the list of lipinski functions and functional groups counters
lipinski_functions = []
functional_groups = []

print("Creating Lipinski functions list...")
for name, val in tqdm(Lipinski.__dict__.items()):
    if(not name.startswith('_') and callable(val)):
        lipinski_functions.append(name)

print("\n")
print("Creating Functional group functions list...")
for name, val in tqdm(Fragments.__dict__.items()):
    if(not name.startswith('_') and callable(val)):
        functional_groups.append(name)

DB_CONFIG = {
    "host": "localhost",
    "port": 33060,  # MySQL X Protocol default port
    "user": "root",
    "password": "secret", # Lol I went through the Security Training
    "schema": "chembl_35"
}

# Ignoring Pandas PerformanceWarning lol
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Disabling all the warning logs lol
RDLogger.DisableLog('rdApp.*')

# Function to count each functional group in a molecule
def fragments(data):
    """Returns data with functional groups as columns."""
    print("Computing Functional groups...")
    for name, val in tqdm(Fragments.__dict__.items()):
        if(not name.startswith('_') and callable(val)):
            data[name] = data['smiles'].apply(lambda x: int(val(Chem.MolFromSmiles(x)) > 0))
    print("\n")

# Function to count the number of Lipinski properties in a molecule
def lipinski(data):
    """Returns data with Lipinski properties as columns."""
    print("Computing Lipinski properties...")
    for name, val in tqdm(Lipinski.__dict__.items()):
        if(not name.startswith('_') and callable(val)):
            data[name] = data['smiles'].apply(lambda x: val(Chem.MolFromSmiles(x)))
    print("\n")

# Function to add a molecule column to the dataframe
def structure(data):
    """Returns data with a molecule column."""
    PandasTools.AddMoleculeColumnToFrame(data, smilesCol='smiles')

# Function to identify functional groups in a molecule
def ifg(data):
    """Returns data with a functional group column."""
    data['ifg'] = data['smiles'].apply(lambda x: identify_functional_groups(Chem.MolFromSmiles(x)))

# Function to read an sdf file
def read_sdf(file, fragments_bool=False, lipinski_bool=False, ifg_bool=False, structure_bool=False, lipophilicity=False, mol_mr=False, exact_mol_wt=True, sample_bool=None):
    """
    Loop through mols objects in SDF file
    
    Parameters:
    file (path): File path for SDF file
    fragments_bool (bool): Set to True for adding functional groups measurement columns in DataFrame
    lipinski_bool (bool): Set to True for adding Lipinski properties columns in DataFrame
    structure_bool (bool): Set to True for adding molecule structure image in DataFrame
    lipophilicity (bool): Set to True for adding lipophilicity values in DataFrame
    mol_mr (bool): Set to True for adding molecular refractivity values in DataFrame
    exact_mol_wt (bool): Set to True for adding exact molecular weight values in DataFrame
    sample_bool (int): Give an input number to sample from the generated DataFrame
    
    Returns:
    pd.DataFrame: Processed dataframe with 'smiles' default column and other additional columns
    """
    suppl = Chem.SDMolSupplier(file)
    
    mols = []
    print("Creating Molecules list from Molecule supplier object...")
    for mol in tqdm(suppl):
        if mol is not None:
            mols.append(mol)
    print("\n")

    # Create a dataframe with the smiles and add other conditional columns
    columns = ['smiles']
    columns.append('exact_mol_wt') if exact_mol_wt else columns
    columns.append('mol_mr') if mol_mr else columns
    columns.append('lipophilicity') if lipophilicity else columns
    data = pd.DataFrame(columns=columns)
    print("Creating primary DataFrame...")
    for mol in tqdm(mols):
        row = {}
        row['smiles'] = Chem.MolToSmiles(mol)
        
        if exact_mol_wt:
            row['exact_mol_wt'] = Descriptors.ExactMolWt(mol)
        if mol_mr:
            row['mol_mr'] = Crippen.MolMR(mol)
        if lipophilicity:
            row['lipophilicity'] = Crippen.MolLogP(mol)

        data.loc[len(data)] = row
    print("\n")
    # data = data.convert_dtypes()
    data = data.drop_duplicates(subset='smiles').copy()
    data = data.convert_dtypes()

    # Remove salts from DataFrame
    print("Removing salts from data...")
    data = data[~data['smiles'].str.contains("\.")]
    print("\n")

    if ifg_bool:
        ifg(data)
    
    if fragments_bool:
        fragments(data)

    if lipinski_bool:
        lipinski(data)

    if structure_bool:
        structure(data)
    
    if sample_bool is not None:
        data = data.sample(sample_bool)
    
    return data.infer_objects()

# Function to convert DataFrame into Series
def to_series(data, drop_columns=None, nlargest_bool=None):
    """Converts a DataFrame into a Series."""
    data = data.infer_objects().select_dtypes(include='number')
    data = data.drop(columns=drop_columns) if drop_columns is not None else data
    data = data.sum().sort_values(ascending=False)
    data = data.nlargest(nlargest_bool) if nlargest_bool is not None else data
    return data

# Function to plot nlargest (if given) functional groups in a dataframe
def plot_bar(data, nlargest=None, title=None, drop_columns=None):
    """Plots a bar graph of the frequency of functional groups/lipinski properties"""
    drop_columns = ['exact_mol_wt'] + drop_columns if drop_columns is not None else ['exact_mol_wt']
    data = to_series(data, drop_columns)
    
    if nlargest is not None:
        data = data.nlargest(nlargest)

    data.plot(kind='bar', title=title)

# Function to compute the essential properties of molecules for rules check
def rules_properties(molecule):
    """Returns a dictionary of all the properties of a Mol object required for drug screening rule testing"""
    properties = {}

    properties['molecular_weight'] = Descriptors.ExactMolWt(molecule)
    properties['logp'] = Descriptors.MolLogP(molecule)
    properties['h_bond_donor'] = Descriptors.NumHDonors(molecule)
    properties['h_bond_acceptors'] = Descriptors.NumHAcceptors(molecule)
    properties['rotatable_bonds'] = Descriptors.NumRotatableBonds(molecule)
    properties['number_of_atoms'] = Chem.rdchem.Mol.GetNumAtoms(molecule)
    properties['molar_refractivity'] = Crippen.MolMR(molecule)
    properties['topological_surface_area_mapping'] = Chem.QED.properties(molecule).PSA
    properties['formal_charge'] = Chem.rdmolops.GetFormalCharge(molecule)
    properties['heavy_atoms'] = Chem.rdchem.Mol.GetNumHeavyAtoms(molecule)
    properties['num_of_rings'] = Chem.rdMolDescriptors.CalcNumRings(molecule)

    return properties

# Function to return whether a molecule follows Lipinski rule of 5
def lipinski_rule_of_5(molecule):
    """Returns whether a molecule follows Lipinski rule of 5."""
    properties = rules_properties(molecule)
    rule1 = properties['molecular_weight'] <= 500
    rule2 = properties['logp'] <= 5
    rule3 = properties['h_bond_donor'] <= 5
    rule4 = properties['h_bond_acceptors'] <= 5
    rule5 = properties['rotatable_bonds'] <= 5

    return rule1 and rule2 and rule3 and rule4 and rule5

# Function to return whether a molecule follows Ghose rule
def ghose_filter(molecule):
    """Returns whether a molecule passes the Ghose filter"""
    properties = rules_properties(molecule)
    rule1 = properties['molecular_weight'] >= 160 and properties['molecular_weight'] <= 480
    rule2 = properties['logp'] >= -0.4 and properties['logp'] <= 5.6
    rule3 = properties['number_of_atoms'] >= 20 and properties['number_of_atoms'] <= 70
    rule4 = properties['molar_refractivity'] >= 40 and properties['molar_refractivity'] <= 130

    return rule1 and rule2 and rule3 and rule4

# Function to return whether a molecule follows Veber rule
def veber_rule(molecule):
    """Returns whether a molecule follows the Veber rule."""
    properties = rules_properties(molecule)
    rule1 = properties['rotatable_bonds'] <= 10
    rule2 = properties['topological_surface_area_mapping'] <= 140

    return rule1 and rule2

# Function to return whether a molecule follows Egan rule
def egan_rule(molecule):
    """Returns whether a molecule follows the Egan rule."""
    properties = rules_properties(molecule)
    rule1 = properties['molecular_weight'] >= 200 and properties['molecular_weight'] <= 600
    rule2 = properties['logp'] >= -0.4 and properties['logp'] <= 5.0
    rule3 = properties['h_bond_donor'] <= 6
    rule4 = properties['h_bond_acceptors'] <= 5 and properties['h_bond_acceptors'] <= 12

    return rule1 and rule2 and rule3 and rule4

# Function to return whether a molecule passes REOS filter
def reos_filter(molecule):
    """Returns whether a molecule passes the REOS filter."""
    properties = rules_properties(molecule)
    rule1 = properties['molecular_weight'] >= 200 and properties['molecular_weight'] <= 500
    rule2 = properties['logp'] >= int(0-5) and properties['logp'] <= 5
    rule3 = properties['h_bond_donor'] >= 0 and properties['h_bond_donor'] <= 5
    rule4 = properties['h_bond_acceptors'] >= 0 and properties['h_bond_acceptors'] <= 10
    rule5 = properties['formal_charge'] >= int(0-2) and properties['formal_charge'] <= 2
    rule6 = properties['rotatable_bonds'] >= 0 and properties['rotatable_bonds'] <= 8
    rule7 = properties['heavy_atoms'] >= 15 and properties['heavy_atoms'] <= 50

    return rule1 and rule2 and rule3 and rule4 and rule5 and rule6 and rule7

# Function to return whether a molecule passes Drug Like filter
def drug_like_filter(molecule):
    """Returns whether a molecule passes the Drug Like filter."""
    properties = rules_properties(molecule)
    rule1 = properties['molecular_weight'] < 400
    rule2 = properties['num_of_rings'] > 0
    rule3 = properties['rotatable_bonds'] < 5
    rule4 = properties['h_bond_donor'] <= 5
    rule5 = properties['h_bond_acceptors'] <= 10
    rule6 = properties['logp'] < 5

    return rule1 and rule2 and rule3 and rule4 and rule5 and rule6

# Function to return whether a molecule passes the rule of 3
def rule_of_three(molecule):
    """Returns whether a molecule passes the rule of 3."""
    properties = rules_properties(molecule)
    rule1 = properties['molecular_weight'] <= 300
    rule2 = properties['logp'] <= 3
    rule3 = properties['h_bond_donor'] <= 3
    rule4 = properties['h_bond_acceptors'] <= 3
    rule5 = properties['rotatable_bonds'] <= 3

    return rule1 and rule2 and rule3 and rule4 and rule5

# Function to return a dictionary of all the rules a molecule passes
def all_rules(molecule):
    """Returns a dictionary of all the rules a molecule passes."""
    rules = {}
    rule1 = rules['lipinski_rule_of_5'] = int(lipinski_rule_of_5(molecule))
    rule2 = rules['ghose_filter'] = int(ghose_filter(molecule))
    rule3 = rules['veber_rule'] = int(veber_rule(molecule))
    rule4 = rules['egan_rule'] = int(egan_rule(molecule))
    rule5 = rules['reos_filter'] = int(reos_filter(molecule))
    rule6 = rules['drug_like_filter'] = int(drug_like_filter(molecule))
    rule7 = rules['rule_of_three'] = int(rule_of_three(molecule))
    rules['passes_all'] = int(rule1 and rule2 and rule3 and rule4 and rule5 and rule6 and rule7)
    rules['fails_all'] = int(not rule1 and not rule2 and not rule3 and not rule4 and not rule5 and not rule6 and not rule7)

    return rules

# Function to plot the number of molecules following each rule
def plot_rules(data, title=None, normalise=False):
    """Function to plot the number of molecules passing each drug screening filter/rule"""
    mols = [Chem.MolFromSmiles(mol) for mol in data['smiles']]
    results = Counter(all_rules(mols[0]))

    print("Computing rules frequency for molecules...\n")
    for i in tqdm(range(1, len(mols))):
        results = results + Counter(all_rules(mols[i]))

    results = pd.Series(dict(results))
    if normalise:
        results = results.apply(lambda x: x*100/len(mols))
    results.plot(kind='bar', title=title)

    return results

# Function to create Butina clusters of molecules
def createButinaClusters(data, radius=2, fpSize=1024, cutoff=0.2):
    """Generates a list of tuples each containing the cluster with the first element in each tuple being the centroid"""
    # Function to create cluster from given fingerprints
    def ClusterFps(fps, cutoff=0.2):
        from rdkit import DataStructs
        from rdkit.ML.Cluster import Butina

        dists = []
        nfps = len(fps)

        for i in tqdm(range(1,nfps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
            dists.extend([1-x for x in sims])

        mol_clusters = Butina.ClusterData(dists,nfps,cutoff,isDistData=True)
        return mol_clusters
    
    mols = []
    # Iterating over rows to get the individual molecules
    print("Iterating through dataset, generating molecules list...")
    for index, row in tqdm(data.iterrows()):
        mol_temp = Chem.MolFromSmiles(row['smiles'])
        mol_type = row['molecule_type']
        mol_temp.SetProp("mol_type", mol_type)
        mols.append(mol_temp)
    
    print("\n")
    fps = []
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
    # Generating Morgan fingerprints for every molecule
    print("Generating molecular fingerprints list...")
    for mol in tqdm(mols):
        fps.append(mfpgen.GetFingerprint(mol))
    
    print("\n")
    # Creating the clusters
    print("Generating Butina clusters...")
    clusters = ClusterFps(fps, cutoff=cutoff)

    # return clusters
    print("\n")
    # Convert the indexes into Molecules
    clusters_list = []
    print("Converting indices back to Molecules list...")
    for tuple in tqdm(clusters):
        mols_list = []
        for i in tuple:
            mols_list.append(mols[i])
        clusters_list.append(mols_list)
    print("\n")

    return clusters_list

# Function to create scatter plot of molecule clusters between two DataFrames
def createSimilarityMatrix(data, radius=2, fpSize=2048):
    """Calculate the distance matrix based on tanimoto similarity index"""
    fps = []
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)

    print("Computing molecule fingerprints...")
    for _, row in tqdm(data.iterrows()):
        fps_i = mfpgen.GetFingerprint(Chem.MolFromSmiles(row['smiles']))
        fps.append(fps_i)
    print("\n")

    print("Creating distance matrix...")
    nfps = len(fps)
    dists = np.zeros([nfps, nfps])
    for i in tqdm(range(nfps)):
        for j in range(i+1, nfps):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            dists[i, j] = sim
            dists[j, i] = sim
    print("\n")

    return dists

# Function to plot 2D graph based on Tanimoto distance matrix
def plot_2d_graph(data, correlationMatrix, colorMap=np.array(['r', 'g', 'b']), column_name='molecule_type'):
    """Plots 2D graph of the given data and corresponding correlation matrix"""
    print("Performing Multi-Dimensional Scaling on distance matrix...\n")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(correlationMatrix)
    plt.figure(figsize=(20, 18))
    categories_list = data[column_name].to_numpy()
    print("Plotting scatter plot using obtained co-ordinates...\n")
    plt.scatter(coords[:, 0], coords[:, 1], c=colorMap[categories_list], alpha=0.7)
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.show()

# Function to generate a list of molecule types for a given Mols cluster list
def generateLegend(cluster, key="mol_type"):
    """Tiny function to generate the corresponding list of molecular metadata for every Mols object in the cluster list"""
    legend = []
    for mol in cluster:
        legend.append(mol.GetProp(key))
    return legend

# Function to output the grid images of molecules for a given clusters group
def printClusterImages(cluster_group, directory_name='output', file_names='grid', filter_thresh=1, display_legend=True, subImgSize=(200,200)):
    """Generates images of each Butina cluster from the list of cluster groups"""
    filtered_clusters = []

    print(f"Filtering out cluster sizes of less than or equal to {filter_thresh}...")
    for cluster in tqdm(cluster_group):
        if len(cluster) > filter_thresh:
            filtered_clusters.append(cluster)
    
    n = len(filtered_clusters)
    print("Creating directory...")
    str = directory_name
    os.mkdir(str)
    print("\n")

    print("Generating cluster images and saving them...")
    for i in tqdm(range(n)):
        path = str+'/'+file_names+f"{i}.png"
        legend = generateLegend(filtered_clusters[i]) if display_legend else None
        p = Draw.MolsToImage(filtered_clusters[i], legends=legend, subImgSize=subImgSize)
        p.save(path)
    print("\n")

# Function to create molecules dictionary a.k.a node data in CSN
def createMolsNodesData(data, smiles_column='smiles', value_column='lipophilicity'):
    """Returns a dictionary of molecules with the smiles being key and a selected value column"""
    """This dictionary is used as node data for creating CSN"""
    smils = data[smiles_column].tolist()
    rdkit_can_smiles = []

    if not(value_column in data.columns):
        raise Exception("Non-existent column in DataFrame. Valid value column required.")

    print("Checking for duplicate values in DataFrame...")
    for smi in tqdm(smils):
        mol = Chem.MolFromSmiles(smi)
        rdkit_can_smiles.append(Chem.MolToSmiles(mol))
    print("\n")

    set_rdkit_can_smiles = set(rdkit_can_smiles)

    if not(len(set_rdkit_can_smiles) == len(rdkit_can_smiles)):
        raise Exception("Duplicate values exist. Clear the DataFrame.")
    
    dat = data.set_index(smiles_column)
    return dat.to_dict('index')

# Function to create a dictionary of combinatorial subsets of given molecule dictionary with an integer key
def createPairsSubsetDict(dict):
    """Creates a dictionary of subsets of all possible molecule pairs"""
    from itertools import combinations

    print("Computing combinations and creating subset list...")
    smis = []
    for key, value in tqdm(dict.items()):
        smis.append(key)
    print("\n")

    smis_subsets = list(combinations(smis, 2))

    print("Creating dictionary with integer key...")
    # create a dictionary, subsets
    subsets = {}
    for i, (smi1,smi2) in tqdm(enumerate(smis_subsets)):
        field = {}
        field["smi1"] = smi1
        subsets[i] = field
        
        field["smi2"] = smi2
        subsets[i] = field

    return subsets

# Function to add Mol objects to subsets dictionary
def addMolToSubsetsDict(dict):
    """Adds a Mols objects values into the dictionary fields"""
    print("Adding Mol objects to dictionary...")
    for key, value in tqdm(dict.items()):
        dict[key].update({'mol1': Chem.MolFromSmiles(value['smi1'])})
        dict[key].update({'mol2': Chem.MolFromSmiles(value['smi2'])})
    print("\n")

# Function to add Tanimoto similarity to subsets dictionary
def addTanimotoSim(dict, radius=2, fpSize=2048):
    """Adds Tanimoto similarity index into the dictionary fields"""
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)

    print("Computing and adding Tanimoto similarity to dictionary...")
    for key, value in tqdm(dict.items()):
        fp1 = mfpgen.GetFingerprint(Chem.MolFromSmiles(value['smi1']))
        fp2 = mfpgen.GetFingerprint(Chem.MolFromSmiles(value['smi2']))
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        dict[key].update({'tan_sim': sim})
    print("\n")

# Function to add Maximum Common Substructure based Tanimoto similarity
# Caution: Use Pool and starmap to use this function
def tc_mcs(mol1,mol2,key):
    """Function to find Tanimoto similarity based on MCS"""
    """To be used with Pool and starmap multithreading ONLY"""
    # get maximum common substructure instance
    mcs = rdFMCS.FindMCS([mol1,mol2],timeout=10) # adding a 10 second timeout

    # get number of common bonds
    mcs_bonds = mcs.numBonds

    # get number of bonds for each
    # default is only heavy atom bonds
    mol1_bonds = mol1.GetNumBonds()
    mol2_bonds = mol2.GetNumBonds()

    # compute MCS-based Tanimoto
    if(mol1_bonds+mol2_bonds-mcs_bonds == 0):
        tan_mcs = 1
    else:
        tan_mcs = mcs_bonds/(mol1_bonds+mol2_bonds-mcs_bonds)
    return key, tan_mcs

# Function to create node graph data using the subsets dictionary
def createGraph(subsets, node_data, attrib_name='molecule_type', filter_thresh=0.68, useMCS=False):
    """Function to create NetworkX graph nodes based on similarity indices"""

    print("Filtering the dictionary...")
    subsets_filtered = {}
    inner_key = 'tan_sim' if not useMCS else 'tan_mcs'
    for key, value in tqdm(subsets.items()):
        if value[inner_key] >= filter_thresh:
            subsets_filtered[key] = value
    print("\n")

    print("Generating nodes in Chemical Space Network graph...")
    G1 = nx.Graph()
    for key, value in tqdm(subsets_filtered.items()):
        G1.add_edge(value['smi1'], value['smi2'], weight=value[inner_key])
    print("\n")

    print("Adding node attributes...")
    for key, value in tqdm(node_data.items()):
        G1.add_node(key, attrib=value[attrib_name])
    print("\n")

    return G1

# Function to create a list of sub-graphs (separately connected) from a given graph object
def createSubGraphsList(graph):
    """Creates a list of sub-graphs for a given NetworkX graph object"""
    connected_graphs = []
    print("Creating list of sub-graphs...")
    for c in tqdm(nx.connected_components(graph)):
        sub_graph = graph.subgraph(c).copy()
        connected_graphs.append(sub_graph)
    print("\n")
    
    return connected_graphs

# Function to generate sub-graph images
def generateSubGraphImages(connected_graphs, color_map=True, folder_path='sub-graphs', file_name='graph', seed=30, k=0.3, node_size=300, edge_colors="black", figsize=(8,8), scale=1, min_nodes_thresh=2):
    """Generates and saves the sub-graph images"""

    print("Creating directory...")
    os.mkdir(folder_path)

    print("Generating images...")
    for i in tqdm(range(len(connected_graphs))):
        if(len(connected_graphs[i]) < min_nodes_thresh):
            continue
        sub = connected_graphs[i]
        file_name_ext = file_name+f"{i}.png"
        file_path = os.path.join(folder_path, file_name_ext)
        pos = nx.spring_layout(sub, seed=seed, k=k, scale=scale)
        fig, ax = plt.subplots(figsize=figsize)
        if color_map:
            color_list = []
            for node in sub.nodes(data=True):
                if node[1]["attrib"] == 'drug':
                    color_list.append('red')
                elif node[1]["attrib"] == 'toxin':
                    color_list.append('blue')
            nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=node_size, node_color=color_list, edgecolors=edge_colors)
        else:
            nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=node_size)
        nx.draw_networkx_edges(sub, pos, edge_color=edge_colors)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
    print("\n")

# Function to plot the CSN using Matplotlib
def plotCSN(graph, k=0.3, seed=40, save_file=False, file_name=None, figsize=(8,8), color_map=None, edge_colors=None, node_size=75):
    """Saves the plot image and generates the plot in notebook"""

    pos = nx.spring_layout(graph, k=k, seed=seed)
    fig, ax = plt.subplots(figsize=figsize)

    if color_map is not None:
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=color_map, edgecolors="black")
    else:
        nx.draw(graph, pos)
    
    if edge_colors is not None:
        nx.draw_networkx_edges(graph, pos, edge_color=edge_colors)

    plt.axis("off")
    
    if save_file:
        file_name = file_name+".png" if file_name is not None else "spring_graph.png"
        plt.savefig(file_name)
    
    plt.show()

# Function to pickle the data and save as pickle file
def pickle_data(data, file_name='data'):
    """Saves the given data in pickle format"""
    file_name = file_name+'.pickle'
    with open(file_name, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

# Function to return image using PIL in png format
def show_png(data):
    """Takes in PIL image from RDKit DrawMols function and returns PNG image"""
    bio = io.BytesIO(data)
    img = Image.open(bio)
    return img

# Function to highlight a given Mols object with a standard color
# adjusted alpha values to make transparent
def highlight_mol(smi,label,color):
    """Takes a smiles string, label and color as input and outputs the molecular structure image with highlighting and label"""
    mol = Chem.MolFromSmiles(smi)
    
    if color == 'darkred':
        rgba = (0.55, 0.0, 0.0, 0.3)       
    elif color == 'red':
        rgba = (1.0, 0.0, 0.0, 0.2)    
    elif color == 'orange':
        rgba = (1.0, 0.65, 0.0, 0.2)        
    elif color == 'yellow':
        rgba = (1.0, 1.0, 0.0, 0.4)        
    elif color == 'green':
        rgba = (0.0, 0.50, 0.0, 0.15)        
    elif color == 'lightblue':
        rgba = (0.68, 0.85, 0.90, 0.5)   
    elif color == 'blue':
        rgba = (0.0, 0.0, 1.0, 0.1)    
    else: # no color
        rgba = (1,1,1,1) 
             
    atoms = []
    for a in mol.GetAtoms():
        atoms.append(a.GetIdx())
    
    bonds = []
    for bond in mol.GetBonds():
        aid1 = atoms[bond.GetBeginAtomIdx()]
        aid2 = atoms[bond.GetEndAtomIdx()]
        bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

    drawer = rdMolDraw2D.MolDraw2DCairo(300,300)
    drawer.drawOptions().fillHighlights=True
    drawer.drawOptions().setHighlightColour((rgba))
    drawer.drawOptions().highlightBondWidthMultiplier=15
    drawer.drawOptions().legendFontSize=60
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, legend=label,highlightAtoms=atoms, highlightBonds=bonds)
    
    mol_png = drawer.GetDrawingText()
    return mol_png

# Function to generate the molecular structure image based CSN with highlighting for a given sub-graph
def structureCSNMap(subgraph, highlighting=True, attrib='attrib', figsize=(12,12), molSize=0.045, seed=30, threshes=[0.9, 0.7], save_file=True, file_path_name='figure'):
    """Creates a chemical structure based CSN graph for a given sub-graph"""
    thick = [(u, v) for (u, v, d) in subgraph.edges(data=True) if d["weight"] >= threshes[0]]
    medium = [(u, v) for (u, v, d) in subgraph.edges(data=True) if threshes[1] < d["weight"] < threshes[0]]
    thin = [(u, v) for (u, v, d) in subgraph.edges(data=True) if d["weight"] <= threshes[1]]

    color = {'drug': 'red', 'toxin': 'blue', 'carcinogen': 'green'}

    pos = nx.spring_layout(subgraph, seed=seed)
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(subgraph, pos=pos, ax=ax, edgelist=thick, width=3, edge_color="lightgrey")
    nx.draw_networkx_edges(subgraph, pos=pos, ax=ax, edgelist=medium, width=1, edge_color="lightgrey")
    nx.draw_networkx_edges(subgraph, pos=pos, ax=ax, edgelist=thin, width=1, alpha=0.5, edge_color="lightgrey", style="dashed")
    ax.axis('off')

    # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
    tr_figure = ax.transData.transform
    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform

    # Select the size of the image (relative to the X axis)
    struct_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * molSize # adjust this value to change size of structure drawings
    struct_center = struct_size / 2.0

    # Add the respective image to each node
    for smi, value in subgraph.nodes.items():
        # draw molecule
        drawer = rdMolDraw2D.MolDraw2DCairo(300,300)
        drawer.drawOptions().clearBackground = False
        drawer.drawOptions().addStereoAnnotation = False
        drawer.DrawMolecule(Chem.MolFromSmiles(smi))
        drawer.FinishDrawing()
        mol = drawer.GetDrawingText()
            
        xf, yf = tr_figure(pos[smi])
        xa, ya = tr_axes((xf, yf))
        # get overlapped axes and plot structure
        a = plt.axes([xa - struct_center, ya - struct_center, struct_size, struct_size])
        if highlighting:
            attribute = value[attrib]
            a.imshow(show_png(highlight_mol(smi=smi, label=attribute, color=color[attribute])))
        else:
            a.imshow(show_png(mol))
        a.axis("off")
    
    if save_file:
        plt.savefig(file_path_name+'.png', bbox_inches='tight')
        plt.close()  
    else:
        plt.show()

# Function to get ChEMBL ID for a given 'SMILES' string
# Uses ChEMBL Web API
def get_chembl_id(smiles):
    """Fetches the ChEMBL ID for a given SMILES string."""
    molecule = new_client.molecule
    try:
        results = molecule.filter(molecule_structures__canonical_smiles=smiles)
        if results:
            return results[0]['molecule_chembl_id']
    except Exception as e:
        print(f"Error fetching ChEMBL ID for {smiles}: {e}")
    return None

# Function to add ChEMBL IDs to the dataframe using Pool.imap
# To be used under main function run block for preventing recursive execution and Out-Of-Memory errors
def add_chembl_ids_to_dataframe(df):
    """Uses Pool.imap for efficient processing with large datasets."""
    smiles_list = df['smiles'].tolist()
    chembl_ids = []
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap(get_chembl_id, smiles_list), total=len(smiles_list), desc="Fetching ChEMBL IDs"):
            chembl_ids.append(result)
    df['CHEMBL'] = chembl_ids
    return df

# Function to check whether standard_type_value value exists for a given ChEMBL ID
def check_chembl_id(chembl_id, standard_type_value):
    """Checks if the given ChEMBL ID has the exact standard_type using CAST."""
    try:
        # Connect to MySQL X Protocol (new connection per process)
        session = mysqlx.get_session(DB_CONFIG)
        db = session.get_schema(DB_CONFIG["schema"])

        # Query using CAST for exact match
        query = f"""
        SELECT EXISTS (
            SELECT 1 FROM activities a
            JOIN molecule_dictionary m ON a.molregno = m.molregno
            WHERE CAST(m.chembl_id AS BINARY) = '{chembl_id}' 
            AND CAST(a.standard_type AS BINARY) = '{standard_type_value}'
        );
        """

        # Execute query
        sql_result = session.sql(query).execute()
        exists = sql_result.fetch_one()[0]  # Fetch result (1 or 0)

        # Close session
        session.close()
        return exists  # 1 if match found, 0 otherwise

    except mysqlx.Error as err:
        print(f"Error: {err}")
        return 0

# Function to count the number of valid ChEMBL IDs present for a value of standard_type_value
def count_matching_chembl_ids(df, standard_type_value):
    """Uses multiprocessing with tqdm to count ChEMBL IDs with the exact standard_type."""
    with Pool(cpu_count()) as pool:  # Use all available CPU cores
        # Use partial to pass the constant argument 'standard_type_value'
        func = partial(check_chembl_id, standard_type_value=standard_type_value)

        # Use tqdm to track progress
        results = list(tqdm(pool.imap_unordered(func, df["CHEMBL"]), 
                            total=len(df), desc="Processing ChEMBL IDs"))  
    
    return sum(results)  # Count occurrences where EXISTS returned 1

# Function to perform SQL query to get ADMET properties data for a given ChEMBL ID
def get_admet_data_from_db(chembl_id, session):
    """Fetches ADMET data for a given ChEMBL ID from the local database using mysqlx."""
    try:
        query = """
        WITH mol_data AS (
            SELECT molregno 
            FROM molecule_dictionary 
            WHERE chembl_id = ?
        )
        SELECT 
            ac.standard_value,
            ac.standard_units,
            ac.standard_type
        FROM activities as ac
        JOIN molecule_dictionary md ON ac.molregno = md.molregno
        WHERE md.molregno IN (SELECT molregno FROM mol_data)
        AND CAST(standard_type AS BINARY) IN ('IC50', 'pKa', 'CL', 'T1/2', 'Solubility', 'Ki', 'Kd', 'MIC', 'LC50', 'Potency', 'Activity', 'EC50', 'PPB', 'Vd')
        AND standard_relation IN ('=');
        """
        stmt = session.sql(query).bind(chembl_id)
        records = stmt.execute().fetch_all()
        return {
            'ChEMBL_ID': chembl_id,
            **{record[2]: float(record[0]) if record[0] is not None else None for record in records}
        }
    except Exception as e:
        print(f"Error fetching ADMET data for {chembl_id}: {e}")
        return {'ChEMBL_ID': chembl_id}

# Function to add ADMET data columns to a given DataFrame containing valid 'SMILES' column
def fetch_admet_for_dataframe(df, db_config):
    """Fetches ADMET data for all ChEMBL IDs in the dataframe from the local database using mysqlx."""
    chembl_ids = df['CHEMBL'].dropna().unique().tolist()
    session = mysqlx.get_session(db_config)
    
    results = []
    for chembl_id in tqdm(chembl_ids, desc="Fetching ADMET Data"):
        results.append(get_admet_data_from_db(chembl_id, session))
    
    session.close()
    return pd.DataFrame(results).set_index('ChEMBL_ID')
"""It is HIGHLY advised to use the tc_mcs function using multiprocessing modules"""
"""The returned graph object can be used to specify color schemes for nodes based on various parameters"""