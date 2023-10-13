import DocClust.algos as algos
import DocClust.metrics as metrics
import DocClust.utils as utils



# Config values. 
csv_dir = 'C:/Users/George Georgariou/Desktop/'
figures_dir = 'C:/Users/George Georgariou/Documents/Visual Studio Code/DocumentClustering/figures/'
parameters_dir = 'C:\\Users\\George Georgariou\\Desktop\\'
local_datasets_path = 'D:\\Datasets\\'
debug = False
reduce_dim = False
nn = 28
min_cluster_size = 20
n_jobs = None
test_dataset = False
limit_corpus_size = 0
random_state = 42

# ------------------------ Datasets - Corpus ------------------------ #

datasets_language = "english"#"english"#

datasets_strings = [
    #"test"
    #"blobs"

    #"20newsgroups",
    "reuters21578",
    "trec",
    #"webace"
    "pubmed4000",
    "classic4",

    #"greek_legal_code_more_500",
    #"greek_legal_code_less_500",
    #"makedonia",
    #"greeksum",
    #"greek_legal_code_400_600",
    #"greek_legal_sum",
    #"ogtd"
]

datasets_en_strings = [
    "test",
    "blobs",
    "20newsgroups",
    "reuters21578",
    "trec",
    "webace",
    "pubmed4000",
    "classic4",
]

datasets_gr_strings = [
    "greek_legal_code_more_600",
    "greek_legal_code_more_500",
    "greek_legal_code_less_500",
    "greek_legal_code_400_600",
    "makedonia",
    "greeksum",
    "greek_legal_sum",
    "ogtd"
]



def datasets_pointers():
    return {
        "20newsgroups": utils.load_dataset_20newsgroups,
        "reuters21578": utils.load_dataset_reuters21578,
        "trec": utils.load_dataset_trec,
        "classic4": utils.load_dataset_classic4,
        "pubmed4000":utils.load_dataset_pubmed4000,
        "webace": utils.load_dataset_webace,
        
        "test": utils.load_dataset_test,
        "blobs": utils.load_dataset_blobs,

        "greek_legal_code_more_500": utils.load_dataset_greek_legal_code_more_500,
        "greek_legal_code_less_500": utils.load_dataset_greek_legal_code_less_500,
        "greek_legal_code_400_600": utils.load_dataset_greek_legal_code_400_600,
        "makedonia": utils.load_dataset_makedonia,
        "greeksum": utils.load_dataset_greeksum,
        "greek_legal_sum": utils.load_dataset_greek_legal_sum,
        "ogtd": utils.load_dataset_greek_ogtd
    }

# ------------------------ Embeddings - Doc Vectors ------------------------ #
vectorizers_strings = [
    "tfidf",

    #"spacy_model_embeddings",
    "sent_transformers_model_embeddings",
    "jina_model_embeddings",

    #"greek_bert_model_embeddings",
    #"sent_transformers_paraph_multi_model_embeddings",
    #"greek_bart_model_embeddings",
    #"greek_spacy_model_embeddings" ,
    #"greek_xlm_roberta_model_embeddings",
]

def vectorizers_pointers():
    return {
        "sent_transformers_model_embeddings": utils.sent_transformers_model_embeddings,
        "jina_model_embeddings": utils.jina_model_embeddings,
        "spacy_model_embeddings": utils.spacy_model_embeddings,
        "tfidf": utils.tfidf,

        "greek_bert_model_embeddings": utils.sent_transformers_model_embeddings,
        "sent_transformers_paraph_multi_model_embeddings": utils.sent_transformers_model_embeddings,
        "greek_bart_model_embeddings": utils.sent_transformers_model_embeddings,
        "greek_spacy_model_embeddings": utils.spacy_model_embeddings,
        "greek_xlm_roberta_model_embeddings": utils.sent_transformers_model_embeddings,
    }


# ------------------------ Clustering Algorithms ------------------------ #
clustering_algorithms_strings = [
     #"kmeans",
     "kmedoids",
     #"agglomerative",
     #"birch",
     #"hdbscan"
]

# Config Clustering algorithm approaches
def clustering_algorithms_parameteres():
    return {
        "kmeans": 
            ['n_clusters', 'algorithm', 'init_centers'],
        "kmedoids":
            ['n_clusters', 'method', 'init_centers'],
        "agglomerative": 
            ['n_clusters', 'compute_full_tree', 'linkage', 'metric'],
        "birch":
            ['n_clusters'],
        "hdbscan":
            ['n_clusters', 'cluster_selection_method']
    }

def clustering_algorithms_arguments(n_clusters):
    return {
        "kmeans": [
            [n_clusters, 'elkan', 'random'],
            [n_clusters, 'lloyd', 'random'],
            [n_clusters, 'elkan', 'k-means++'],
            [n_clusters, 'lloyd', 'k-means++']
        ],
        "kmedoids":[
            #[n_clusters, 'pam', 'build'],
            #[n_clusters, 'pam', 'k-medoids++'],
            #[n_clusters, 'alternate', 'build'],
            #[n_clusters, 'alternate', 'k-medoids++']
        ],
        "agglomerative": [
            [n_clusters, True, 'ward', 'euclidean'],
            [n_clusters, True, 'complete', 'cosine'],
            [n_clusters, False, 'ward', 'euclidean'],
            [n_clusters, False, 'complete', 'cosine']
        ],
        "birch":[
            [n_clusters]
        ],
        "hdbscan":[
            #[n_clusters, 'eom'],
            [n_clusters, 'leaf']
        ]
    }
        
def clustering_algorithms_pointers():
    return {
        "kmeans": algos.kmeans,
        "kmedoids": algos.kmedoids,
        "agglomerative": algos.agglomerative,
        "birch": algos.birch,
        "hdbscan": algos.hdbscan
    }



# ------------------------ Ext Evaluation Metrics ------------------------ #
evaluation_metrics_strings = [
    "accuracy",
    "mutual_information",
    "adjusted_mutual_information",
    "rand_index",
    "adjusted_rand_index",
    "fowlkes_mallows_index",
    "v_measure_index",
    "homogenity",
    "completeness"
]

def evaluation_metrics_pointers():
    return {
        "accuracy": metrics.accuracy,
        "mutual_information": metrics.mutual_information,
        "adjusted_mutual_information": metrics.adjusted_mutual_information,
        "rand_index": metrics.rand_index,
        "adjusted_rand_index": metrics.adjusted_rand_index,
        "fowlkes_mallows_index": metrics.fowlkes_mallows_index,
        "v_measure_index": metrics.v_measure_index,
        "homogenity": metrics.homogenity,
        "completeness": metrics.completeness
    }


greek_stop_words = set(
"""
αδιάκοπα αι ακόμα ακόμη ακριβώς άλλα αλλά αλλαχού άλλες άλλη άλλην
άλλης αλλιώς αλλιώτικα άλλο άλλοι αλλοιώς αλλοιώτικα άλλον άλλος άλλοτε αλλού
άλλους άλλων άμα άμεσα αμέσως αν ανά ανάμεσα αναμεταξύ άνευ αντί αντίπερα αντίς
άνω ανωτέρω άξαφνα απ απέναντι από απόψε άρα άραγε αρκετά αρκετές
αρχικά ας αύριο αυτά αυτές αυτή αυτήν αυτής αυτό αυτοί αυτόν αυτός αυτού αυτούς
αυτών αφότου αφού

βέβαια βεβαιότατα

γι για γιατί γρήγορα γύρω

δα δε δείνα δεν δεξιά δήθεν δηλαδή δι δια διαρκώς δικά δικό δικοί δικός δικού
δικούς διόλου δίπλα δίχως

εάν εαυτό εαυτόν εαυτού εαυτούς εαυτών έγκαιρα εγκαίρως εγώ εδώ ειδεμή είθε είμαι
είμαστε είναι εις είσαι είσαστε είστε είτε είχα είχαμε είχαν είχατε είχε είχες έκαστα
έκαστες έκαστη έκαστην έκαστης έκαστο έκαστοι έκαστον έκαστος εκάστου εκάστους εκάστων
εκεί εκείνα εκείνες εκείνη εκείνην εκείνης εκείνο εκείνοι εκείνον εκείνος εκείνου
εκείνους εκείνων εκτός εμάς εμείς εμένα εμπρός εν ένα έναν ένας ενός εντελώς εντός
εναντίον  εξής  εξαιτίας  επιπλέον επόμενη εντωμεταξύ ενώ εξ έξαφνα εξήσ εξίσου έξω επάνω
επειδή έπειτα επί επίσης επομένως εσάς εσείς εσένα έστω εσύ ετέρα ετέραι ετέρας έτερες
έτερη έτερης έτερο έτεροι έτερον έτερος ετέρου έτερους ετέρων ετούτα ετούτες ετούτη ετούτην
ετούτης ετούτο ετούτοι ετούτον ετούτος ετούτου ετούτους ετούτων έτσι εύγε ευθύς ευτυχώς εφεξής
έχει έχεις έχετε έχομε έχουμε έχουν εχτές έχω έως έγιναν  έγινε  έκανε  έξι  έχοντας

η ήδη ήμασταν ήμαστε ήμουν ήσασταν ήσαστε ήσουν ήταν ήτανε ήτοι ήττον

θα

ι ιδία ίδια ίδιαν ιδίας ίδιες ίδιο ίδιοι ίδιον ίδιοσ ίδιος ιδίου ίδιους ίδιων ιδίως ιι ιιι
ίσαμε ίσια ίσως

κάθε καθεμία καθεμίας καθένα καθένας καθενός καθετί καθόλου καθώς και κακά κακώς καλά
καλώς καμία καμίαν καμίας κάμποσα κάμποσες κάμποση κάμποσην κάμποσης κάμποσο κάμποσοι
κάμποσον κάμποσος κάμποσου κάμποσους κάμποσων κανείς κάνεν κανένα κανέναν κανένας
κανενός κάποια κάποιαν κάποιας κάποιες κάποιο κάποιοι κάποιον κάποιος κάποιου κάποιους
κάποιων κάποτε κάπου κάπως κατ κατά κάτι κατιτί κατόπιν κάτω κιόλας κλπ κοντά κτλ κυρίως

λιγάκι λίγο λιγότερο λόγω λοιπά λοιπόν

μα μαζί μακάρι μακρυά μάλιστα μάλλον μας με μεθαύριο μείον μέλει μέλλεται μεμιάς μεν
μερικά μερικές μερικοί μερικούς μερικών μέσα μετ μετά μεταξύ μέχρι μη μήδε μην μήπως
μήτε μια μιαν μιας μόλις μολονότι μονάχα μόνες μόνη μόνην μόνης μόνο μόνοι μονομιάς
μόνος μόνου μόνους μόνων μου μπορεί μπορούν μπρος μέσω  μία  μεσώ

να ναι νωρίς

ξανά ξαφνικά

ο οι όλα όλες όλη όλην όλης όλο ολόγυρα όλοι όλον ολονέν όλος ολότελα όλου όλους όλων
όλως ολωσδιόλου όμως όποια οποιαδήποτε οποίαν οποιανδήποτε οποίας οποίος οποιασδήποτε οποιδήποτε
όποιες οποιεσδήποτε όποιο οποιοδηήποτε όποιοι όποιον οποιονδήποτε όποιος οποιοσδήποτε
οποίου οποιουδήποτε οποίους οποιουσδήποτε οποίων οποιωνδήποτε όποτε οποτεδήποτε όπου
οπουδήποτε όπως ορισμένα ορισμένες ορισμένων ορισμένως όσα οσαδήποτε όσες οσεσδήποτε
όση οσηδήποτε όσην οσηνδήποτε όσης οσησδήποτε όσο οσοδήποτε όσοι οσοιδήποτε όσον οσονδήποτε
όσος οσοσδήποτε όσου οσουδήποτε όσους οσουσδήποτε όσων οσωνδήποτε όταν ότι οτιδήποτε
ότου ου ουδέ ούτε όχι οποία  οποίες  οποίο  οποίοι  οπότε  ος

πάνω  παρά  περί  πολλά  πολλές  πολλοί  πολλούς  που  πρώτα  πρώτες  πρώτη  πρώτο  πρώτος  πως
πάλι πάντα πάντοτε παντού πάντως πάρα πέρα πέρι περίπου περισσότερο πέρσι πέρυσι πια πιθανόν
πιο πίσω πλάι πλέον πλην ποιά ποιάν ποιάς ποιές ποιό ποιοί ποιόν ποιός ποιού ποιούς
ποιών πολύ πόσες πόση πόσην πόσης πόσοι πόσος πόσους πότε ποτέ πού πούθε πουθενά πρέπει
πριν προ προκειμένου πρόκειται πρόπερσι προς προτού προχθές προχτές πρωτύτερα πώς

σαν σας σε σεις σου στα στη στην στης στις στο στον στου στους στων συγχρόνως
συν συνάμα συνεπώς συχνάς συχνές συχνή συχνήν συχνής συχνό συχνοί συχνόν
συχνός συχνού συχνούς συχνών συχνώς σχεδόν

τα τάδε ταύτα ταύτες ταύτη ταύτην ταύτης ταύτοταύτον ταύτος ταύτου ταύτων τάχα τάχατε
τελευταία  τελευταίο  τελευταίος  τού  τρία  τρίτη  τρεις τελικά τελικώς τες τέτοια τέτοιαν
τέτοιας τέτοιες τέτοιο τέτοιοι τέτοιον τέτοιος τέτοιου
τέτοιους τέτοιων τη την της τι τίποτα τίποτε τις το τοι τον τοσ τόσα τόσες τόση τόσην
τόσης τόσο τόσοι τόσον τόσος τόσου τόσους τόσων τότε του τουλάχιστο τουλάχιστον τους τούς τούτα
τούτες τούτη τούτην τούτης τούτο τούτοι τούτοις τούτον τούτος τούτου τούτους τούτων τυχόν
των τώρα

υπ υπέρ υπό υπόψη υπόψιν ύστερα

χωρίς χωριστά

ω ως ωσάν ωσότου ώσπου ώστε ωστόσο ωχ
""".split()
)