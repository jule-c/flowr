import os

try:
    ROOT_DIRPATH = "/hpfs/userws/cremej01/projects/genbench3d"
    if not os.path.exists(ROOT_DIRPATH):
        os.mkdir(ROOT_DIRPATH)
except:
    ROOT_DIRPATH = "/sharedhome/cremej01/workspace/genbench3d"
    if not os.path.exists(ROOT_DIRPATH):
        os.mkdir(ROOT_DIRPATH)

BENCHMARK_DIRNAME = "paper_reproduce"
BENCHMARK_DIRPATH = os.path.join(ROOT_DIRPATH, BENCHMARK_DIRNAME)
if not os.path.exists(BENCHMARK_DIRPATH):
    os.mkdir(BENCHMARK_DIRPATH)

DATA_DIRNAME = "data"
DATA_DIRPATH = os.path.join(BENCHMARK_DIRPATH, DATA_DIRNAME)
if not os.path.exists(DATA_DIRPATH):
    os.mkdir(DATA_DIRPATH)

BIO_CONF_DIRNAME = "generated_molecules"
GEN_CONF_DIRNAME = "generated_molecules"

GEOM_DIRNAME = "GEOM"
GEOM_DIRPATH = os.path.join(ROOT_DIRPATH, GEOM_DIRNAME)

GEOM_RDKIT_DIRNAME = "rdkit_folder"
GEOM_RDKIT_DIRPATH = os.path.join(GEOM_DIRPATH, GEOM_RDKIT_DIRNAME)

GEOM_DRUGS_DIRNAME = "drugs"
GEOM_DRUGS_SUMMARY_FILENAME = "summary_drugs.json"
GEOM_DRUGS_SUMMARY_FILEPATH = os.path.join(
    ROOT_DIRPATH, GEOM_DIRNAME, GEOM_RDKIT_DIRNAME, GEOM_DRUGS_SUMMARY_FILENAME
)

PREPARE_RECEPTOR_BIN_PATH = (
    "/hpfs/userws/cremej01/projects/tools/ADFRsuite-1.0/bin/prepare_receptor"
)

# VINA_DIRPATH = '/home/bb596/vina/'
# VINA_URL = 'https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.4/vina_1.2.4_linux_x86_64'
# VINA_BIN_FILENAME = VINA_URL.split('/')[0]
# VINA_BIN_FILEPATH = os.path.join(VINA_DIRPATH, VINA_BIN_FILENAME)

LIGANDEXPO_FILENAME = "Components-smiles-stereo-cactvs.smi"
BASE_LIGANDEXPO_URL = "http://ligand-expo.rcsb.org/dictionaries"
LIGANDEXPO_URL = f"{BASE_LIGANDEXPO_URL}/{LIGANDEXPO_FILENAME}"
LIGANDEXPO_DIRNAME = "LigandExpo"
LIGANDEXPO_DIRPATH = os.path.join(ROOT_DIRPATH, LIGANDEXPO_DIRNAME)
if not os.path.exists(LIGANDEXPO_DIRPATH):
    os.mkdir(LIGANDEXPO_DIRPATH)
LIGANDEXPO_FILEPATH = os.path.join(LIGANDEXPO_DIRPATH, LIGANDEXPO_FILENAME)

# These URL needs to be provided by the user, as PDBbind is under license
# and requires to be logged in. The "Cloud CDN" links are faster than "Local Download" links
# PDBBIND_GENERAL_URL: str = 'PDBbind_v2020_other_PL'
PDBBIND_GENERAL_URL = "https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_other_PL.tar.gz"
if PDBBIND_GENERAL_URL is None:
    raise Exception(
        """PDBBIND_GENERAL_URL needs to be given, 
                    go to http://www.pdbbind.org.cn/download.php, 
                    and find the links to the general set"""
    )

# PDBBIND_REFINED_URL: str = 'PDBbind_v2020_refined'
PDBBIND_REFINED_URL = (
    "https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/PDBbind_v2020_refined.tar.gz"
)
if PDBBIND_REFINED_URL is None:
    raise Exception(
        """PDBBIND_REFINED_URL needs to be given, 
                    go to http://www.pdbbind.org.cn/download.php, 
                    and find the links to the refined set"""
    )

PDBBIND_CORE_URL = (
    "https://pdbbind.oss-cn-hangzhou.aliyuncs.com/download/CASF-2016.tar.gz"
)
if PDBBIND_CORE_URL is None:
    raise Exception(
        """PDBBIND_CORE_URL needs to be given, 
                    go to http://www.pdbbind.org.cn/casf.php, 
                    and find the links to the core set"""
    )

PDBBIND_DIRNAME = "PDBbind"
PDBBIND_DIRPATH = os.path.join(ROOT_DIRPATH, PDBBIND_DIRNAME)
if not os.path.exists(PDBBIND_DIRPATH):
    os.mkdir(PDBBIND_DIRPATH)

PDBBIND_GENERAL_TARGZ_FILENAME = PDBBIND_GENERAL_URL.split("/")[-1]
PDBBIND_GENERAL_TARGZ_FILEPATH = os.path.join(
    PDBBIND_DIRPATH, PDBBIND_GENERAL_TARGZ_FILENAME
)
PDBBIND_GENERAL_DIRNAME = "general"
PDBBIND_GENERAL_DIRPATH = os.path.join(PDBBIND_DIRPATH, PDBBIND_GENERAL_DIRNAME)
# if not os.path.exists(PDBBIND_GENERAL_DIRPATH):
#     os.mkdir(PDBBIND_GENERAL_DIRPATH)

PDBBIND_REFINED_TARGZ_FILENAME = PDBBIND_REFINED_URL.split("/")[-1]
PDBBIND_REFINED_TARGZ_FILEPATH = os.path.join(
    PDBBIND_DIRPATH, PDBBIND_REFINED_TARGZ_FILENAME
)
PDBBIND_REFINED_DIRNAME = "refined"
PDBBIND_REFINED_DIRPATH = os.path.join(PDBBIND_DIRPATH, PDBBIND_REFINED_DIRNAME)
# if not os.path.exists(PDBBIND_REFINED_DIRPATH):
#     os.mkdir(PDBBIND_REFINED_DIRPATH)

PDBBIND_CORE_TARGZ_FILENAME = PDBBIND_CORE_URL.split("/")[-1]
PDBBIND_CORE_TARGZ_FILEPATH = os.path.join(PDBBIND_DIRPATH, PDBBIND_CORE_TARGZ_FILENAME)
PDBBIND_CORE_DIRNAME = "core"
PDBBIND_CORE_DIRPATH = os.path.join(PDBBIND_DIRPATH, PDBBIND_CORE_DIRNAME)

EDIA_BIN_DIRPATH = (
    "/home/bb596/genbench3d/ediascorer_1.1.0_ubuntu-16.04-64bit/ediascorer_1.1.0/"
)
EDIA_BIN_FILENAME = "ediascorer"
EDIA_BIN_FILEPATH = os.path.join(EDIA_BIN_DIRPATH, EDIA_BIN_FILENAME)
EDIA_LICENSE_FILENAME = "license20230804-593-34k9un.txt"
EDIA_LICENSE_FILEPATH = os.path.join(EDIA_BIN_DIRPATH, EDIA_LICENSE_FILENAME)
EDIA_DATA_DIRNAME = "EDIA_results"
EDIA_DATA_DIRPATH = os.path.join(ROOT_DIRPATH, EDIA_DATA_DIRNAME)


PDB_REDO_DIRNAME = "PDB-REDO"
PDB_REDO_DIRPATH = os.path.join(ROOT_DIRPATH, PDB_REDO_DIRNAME)

CCP4_DIRNAME = "CCP4"
CCP4_DIRPATH = os.path.join(ROOT_DIRPATH, CCP4_DIRNAME)

LIGBOUNDCONF_MINIMIZED_FILENAME = "S2_LigBoundConf_minimized.sdf"
LIGBOUNDCONF_MINIMIZED_FILEPATH = os.path.join(
    ROOT_DIRPATH, LIGBOUNDCONF_MINIMIZED_FILENAME
)

DEFAULT_TFD_THRESHOLD = 0.2

SCHRODINGER_PATH = "/usr/local/shared/schrodinger/current/"
GLIDE_OUTPUT_DIRPATH = "/hpfs/userws/cremej01/projects/glide_working_dir/"

CSD_DRUG_SUBSET_PATH = os.path.join(ROOT_DIRPATH, "CSD_Drug_Subset.gcd")

CROSSDOCKED_PATH = "/hpfs/projects/mlcs/mlhub/e3moldiffusion/data/crossdocked"
CROSSDOCKED_DATA_PATH = os.path.join(CROSSDOCKED_PATH, "crossdocked_v1.1_rmsd1.0")
CROSSDOCKED_POCKET10_PATH = os.path.join(CROSSDOCKED_PATH, "crossdocked_pocket10/")
CROSSDOCKED_SPLITS_PT_PATH = os.path.join(CROSSDOCKED_PATH, "split_by_name.pt")
CROSSDOCKED_SPLITS_P_PATH = os.path.join(CROSSDOCKED_PATH, "split_by_name.p")

MINIMIZED_DIRNAME = "minimized/"
MINIMIZED_DIRPATH = os.path.join(BENCHMARK_DIRPATH, MINIMIZED_DIRNAME)

MIN_PATTERN_VALUES = 50

Q_VALUE_THRESHOLD = 0.001
CLASH_SAFETY_RATIO = 0.75
MAX_RING_PLANE_DISTANCE = 0.1
CONSIDER_HYDROGENS = False
INCLUDE_TORSIONS = False
ADD_MINIMIZED_DOCKING_SCORES = True
OVERWRITE = False

TARGETDIFF_RESULTS_FILEPATH = os.path.join(
    BENCHMARK_DIRPATH, "targetdiff/targetdiff_vina_docked.p"
)
THREEDSBDD_GEN_DIRPATH = os.path.join(BENCHMARK_DIRPATH, "AR/test_set/")
POCKET2MOL_GEN_DIRPATH = os.path.join(BENCHMARK_DIRPATH, "Pocket2Mol/test_set/")
DIFFSBDD_GEN_DIRPATH = os.path.join(
    BENCHMARK_DIRPATH, "DiffSBDD/crossdocked_fullatom_joint/"
)
LIGAN_GEN_DIRPATH = os.path.join(BENCHMARK_DIRPATH, "LiGAN/molecules/")
RESGEN_GEN_DIRPATH = os.path.join(BENCHMARK_DIRPATH, "ResGen/test_set/")
