Search.setIndex({"docnames": ["developer/design/considerations", "developer/design/creation", "developer/design/distribution", "developer/design/index", "developer/design/motivation", "developer/design/relationships", "developer/design/shape", "developer/design/sparsity", "developer/design/tensor_stack", "developer/index", "index", "sparse_maps/background", "sparse_maps/design", "sparse_maps/from_sparse_map", "sparse_maps/index", "terminology"], "filenames": ["developer/design/considerations.rst", "developer/design/creation.rst", "developer/design/distribution.rst", "developer/design/index.rst", "developer/design/motivation.rst", "developer/design/relationships.rst", "developer/design/shape.rst", "developer/design/sparsity.rst", "developer/design/tensor_stack.rst", "developer/index.rst", "index.rst", "sparse_maps/background.rst", "sparse_maps/design.rst", "sparse_maps/from_sparse_map.rst", "sparse_maps/index.rst", "terminology.rst"], "titles": ["TensorWrapper Considerations", "Creating a Tensor", "Distribution Design", "Design Documentation", "Motivating TensorWrapper", "Tensor Relationships Design", "Tensor Shape Design", "Tensor Sparsity Design", "Proposed Tensor Stack", "Developer Documentation", "TensorWrapper", "SparseMap Background", "Sparse Map Library Design", "Sparsifying a Tensor", "Sparse Maps Sublibrary", "TensorWrapper Terminology"], "terms": {"tensor": [0, 3, 10, 11, 14], "ar": [0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 13], "domain": [0, 4, 11, 13, 14], "specif": [0, 1, 4, 7], "languag": [0, 4], "dsl": [0, 3, 6, 8], "physic": [0, 1, 4, 8, 11], "sinc": [0, 4, 5, 13], "nearli": 0, "everi": [0, 2], "law": 0, "i": [0, 1, 3, 4, 8, 11, 12, 13, 15], "succinctli": [0, 4], "summar": [0, 1], "equat": [0, 4, 8], "That": [0, 4, 5], "said": [0, 5], "naiv": 0, "creat": [0, 3, 13], "arrai": [0, 4, 6], "float": [0, 1, 6], "point": [0, 1, 4, 6, 8, 12, 13], "valu": [0, 4, 5, 6, 7], "subject": [0, 8], "them": 0, "mathemat": [0, 1, 5], "oper": [0, 1, 4, 5, 7, 8, 12], "impli": 0, "often": [0, 5], "leav": 0, "much": 0, "tabl": 0, "nonetheless": 0, "we": [0, 1, 3, 4, 8, 11, 12, 13], "argu": 0, "have": [0, 4, 5, 7, 8, 12, 13], "base": [0, 4, 6, 7, 8], "import": [0, 5], "code": [0, 4, 8, 13], "becaus": [0, 1, 4, 8], "facilit": 0, "translat": 0, "theori": [0, 4, 8], "encapsul": [0, 12], "optim": [0, 4, 8], "easier": 0, "read": 0, "ration": 0, "about": [0, 6], "The": [0, 1, 2, 4, 7, 8, 12, 13, 15], "follow": 0, "subsect": 0, "our": [0, 4, 7, 8, 13], "tw": 0, "must": [0, 1, 2, 8, 13], "contend": 0, "process": [0, 2, 3, 4, 6, 7], "go": [0, 8, 13], "from": [0, 4, 8, 11, 12, 13], "express": [0, 1, 4, 8], "implement": [0, 4, 6, 7, 8, 12], "written": [0, 4, 8], "should": [0, 8], "competit": 0, "hand": [0, 8], "roll": 0, "when": [0, 5], "appropri": 0, "thi": [0, 1, 4, 6, 7, 8, 12, 13], "requir": [0, 1, 6, 8, 12, 13, 15], "take": [0, 4, 13], "advantag": 0, "hardwar": [0, 2, 8], "intrins": 0, "acceler": [0, 4], "memori": [0, 1, 2, 4, 5, 6, 12], "On": 0, "fly": [0, 2], "recomput": 0, "v": [0, 2, 11, 12], "store": [0, 1, 2, 4, 5, 7, 8, 11, 12], "liter": [0, 1], "data": [0, 12, 13], "layout": [0, 1, 6], "distribut": [0, 1, 3, 6], "replic": [0, 2], "order": [0, 1, 5], "factor": [0, 12], "common": [0, 12], "intermedi": [0, 4, 8], "spars": [0, 2, 5, 7, 10, 11], "natur": [0, 4], "symmetri": [0, 1, 5], "e": [0, 2, 4, 7, 8, 12, 13], "g": [0, 2, 7, 8, 12], "permut": [0, 1, 5, 13], "antisymmetr": [0, 1], "easi": [0, 6, 8], "arbitrari": 0, "einstein": 0, "notat": [0, 11], "basic": [0, 12], "math": 0, "slice": [0, 13], "algorithm": [0, 4, 6, 14], "movement": 0, "ideal": 0, "tell": [0, 2], "do": [0, 1, 3, 8], "doe": [0, 4], "performantli": 0, "parallelzon": 0, "provid": [0, 8, 13], "view": [0, 8], "runtim": [0, 8], "us": [0, 1, 4, 11, 12, 13], "reason": [0, 1, 6, 12, 13], "treat": 0, "interfac": 0, "allow": 0, "mai": [0, 6, 8, 13], "mechan": [0, 8], "overrid": 0, "underli": [0, 4, 8], "entri": [1, 8], "tensorwrapp": [1, 3, 6, 7, 8], "break": 1, "creation": 1, "two": [1, 4, 7, 13, 15], "piec": [1, 13], "fill": [1, 13], "object": [1, 2, 4, 7, 12, 13], "relev": [1, 5], "fig": [1, 8], "2": [1, 4, 15], "need": [1, 3, 4, 8, 12], "befor": 1, "can": [1, 2, 4, 5, 6, 7, 8, 12], "user": [1, 3, 7, 8, 13], "": [1, 3, 4, 5], "properti": 1, "most": [1, 4, 5, 6, 7, 8, 13], "fundament": [1, 7], "shape": [1, 3, 7], "includ": [1, 8], "number": [1, 4, 6, 8, 15], "mode": [1, 5, 6, 7, 13], "extent": [1, 6], "each": [1, 2, 4, 6, 11, 12], "tile": [1, 2, 6, 7, 12, 13], "substructur": 1, "element": [1, 2, 4, 5, 6, 7, 12, 13, 15], "For": [1, 2, 5, 6, 7, 13, 15], "more": [1, 2, 4, 5, 6, 7, 13], "see": 1, "design": [1, 9, 10, 14], "strictli": [1, 4, 6], "speak": [1, 11], "onli": [1, 2, 5], "inform": [1, 6, 8], "make": 1, "all": [1, 4, 8, 12, 13], "other": [1, 4, 5, 8, 13], "perform": [1, 3, 5, 6, 8], "arguabl": [1, 5, 6], "next": 1, "what": [1, 3, 8], "call": [1, 15], "relationship": [1, 3], "whether": [1, 7], "symmetr": [1, 5], "respect": [1, 4], "index": [1, 4, 12, 13], "gener": [1, 2, 4, 5, 8], "properli": [1, 7], "account": 1, "reduc": [1, 13], "consumpt": 1, "compon": [1, 2, 7], "In": [1, 4, 5, 7, 8, 11, 13], "mani": [1, 7, 8], "exhibit": [1, 7], "sparsiti": [1, 3, 5, 8, 11, 12], "which": [1, 2, 4, 7, 8, 11, 12, 13], "approxim": 1, "zero": [1, 5, 7], "implicitli": [1, 7, 12], "greatli": 1, "overhead": 1, "similarli": 1, "fact": 1, "block": 1, "plu": 1, "non": [1, 2, 5, 7], "time": [1, 5, 7, 8], "speed": 1, "up": [1, 4, 13], "evalu": 1, "respons": [1, 7], "describ": 1, "last": [1, 8], "locat": 1, "fall": [1, 8], "state": [1, 2, 8], "final": 1, "one": [1, 2, 5, 13], "how": [1, 4, 8, 12], "info": 1, "total": [1, 15], "uniqu": 1, "null": 2, "contain": [2, 5, 6, 7, 13], "These": [2, 6, 13], "live": [2, 8], "somewher": 2, "u": [2, 4, 11, 12, 13], "where": [2, 4, 6, 8], "particular": 2, "small": 2, "core": 2, "vector": [2, 4, 11, 15], "format": 2, "row": [2, 15], "column": [2, 15], "major": 2, "an": [2, 4, 5, 6, 8, 13, 15], "superflu": 2, "larger": 2, "don": 2, "t": [2, 4, 13], "want": [2, 4, 8, 13], "keep": 2, "track": 2, "complic": [2, 4, 8], "own": 2, "some": [2, 5, 6], "lazi": 2, "eager": 2, "built": 2, "immedi": 2, "delet": 2, "theoret": 2, "mix": 2, "1st": 2, "gpu": [2, 4], "topic": 3, "section": [3, 8], "captur": [3, 6], "motiv": [3, 8], "why": 3, "suffici": 3, "consider": [3, 8, 14], "experi": [3, 8], "propos": 3, "stack": 3, "specifi": [3, 15], "detail": [3, 6, 13], "construct": 3, "n": [4, 6], "defin": [4, 7, 13], "multi": 4, "dimension": [4, 6], "usual": 4, "scalar": [4, 6, 15], "practic": [4, 11], "mean": [4, 7, 8, 13], "less": [4, 13], "ani": [4, 6, 8], "quantiti": 4, "accord": 4, "given": [4, 5, 8, 13], "preval": 4, "seen": 4, "term": [4, 6, 8, 11], "As": [4, 8], "perfect": 4, "exampl": [4, 5], "consid": 4, "energi": 4, "harmon": 4, "oscil": 4, "frac": 4, "1": [4, 8, 13, 15], "sum_": 4, "k_i": 4, "left": 4, "r_i": 4, "r": 4, "0": [4, 13, 15], "_i": 4, "right": 4, "forc": 4, "constant": 4, "displac": 4, "th": [4, 13], "dimens": 4, "simpl": [4, 8], "few": 4, "peopl": 4, "would": [4, 12], "actual": [4, 5, 8, 12], "bring": 4, "comput": [4, 5, 6], "someth": 4, "like": [4, 12], "doubl": 4, "auto": [4, 8], "dr": 4, "r0": 4, "k": 4, "return": [4, 13], "5": 4, "thei": [4, 5, 8], "d": 4, "loop": [4, 13], "pseudo": 4, "version": 4, "dr2": 4, "ignor": 4, "compil": [4, 8], "face": 4, "count": 4, "same": [4, 12, 13], "subtract": 4, "squar": 4, "dot": 4, "product": 4, "primari": 4, "differ": [4, 5, 12], "addit": [4, 12], "while": [4, 8], "inferr": 4, "present": [4, 13], "here": 4, "assum": [4, 8, 13], "akin": 4, "also": [4, 5, 15], "quit": 4, "abl": [4, 6, 8, 13], "better": 4, "so": [4, 8], "produc": [4, 7], "overal": 4, "short": 4, "answer": 4, "situat": [4, 7], "By": [4, 12], "intent": 4, "punt": 4, "backend": 4, "modern": 4, "orient": 4, "program": 4, "techniqu": 4, "disconnect": 4, "between": [4, 8], "appar": 4, "api": [4, 6, 7, 10, 12], "thing": [4, 14], "being": [4, 8], "just": 4, "look": 4, "ha": [4, 8, 15], "extra": 4, "usag": 4, "doesn": 4, "realli": [4, 6], "refer": [4, 7], "could": 4, "identifi": 4, "furthermor": [4, 7], "entir": 4, "possibl": [4, 5, 6], "parallel": 4, "howev": 4, "rewritten": 4, "thread": 4, "potenti": [4, 5, 7], "lot": 4, "work": [4, 7, 13], "multipl": [4, 5, 12], "librari": [4, 8, 10, 14], "out": [4, 5, 6, 8, 12, 13], "find": 4, "strive": 4, "achiev": 4, "full": [4, 8, 13], "featur": [4, 8], "you": 4, "re": 4, "willing": 4, "drop": 4, "precis": 4, "interest": [4, 7], "It": [5, 12], "word": [5, 13], "linearli": 5, "independ": [5, 13], "paramet": [5, 8], "know": 5, "m": 5, "matrix": [5, 15], "A": [5, 6, 11, 15], "its": [5, 11, 15], "satisfi": 5, "a_": 5, "ij": 5, "ji": 5, "turn": 5, "diagon": 5, "half": 5, "off": 5, "depend": [5, 13], "sai": [5, 13], "exist": [5, 8, 12, 13], "among": 5, "those": [5, 8], "intention": 5, "avoid": [5, 12], "spatial": 5, "connot": 5, "If": [5, 7, 13], "both": [5, 13], "wast": 5, "redund": 5, "occur": [5, 8, 13], "obligatori": 5, "caveat": 5, "name": 5, "standard": 5, "access": 5, "pattern": 5, "ruin": 5, "thu": [5, 7, 8, 13], "sometim": 5, "benefici": 5, "antisymmetri": 5, "list": 5, "hermitian": 5, "anti": 5, "complex": 5, "cyclic": 5, "abov": 5, "three": 5, "indic": [5, 7, 12, 13, 15], "basi": 5, "function": [5, 12, 13], "scope": 5, "manifest": 5, "map": [5, 6, 7, 8, 10, 11, 13], "offset": [5, 13], "anoth": 5, "page": [6, 7, 13], "class": [6, 9, 13, 14], "purpos": [6, 7], "noth": 6, "than": [6, 8, 13], "bunch": 6, "typic": [6, 7], "arrang": 6, "rectangular": 6, "repres": 6, "primit": 6, "without": [6, 12], "even": 6, "begin": 6, "lai": 6, "interact": [6, 7], "larg": [6, 8], "intern": 6, "vice": 6, "versa": 6, "sort": 6, "restrict": 6, "recurs": [6, 7], "thought": 6, "field": 6, "get": [6, 13], "document": [7, 10], "context": 7, "possess": 7, "effect": 7, "exactli": 7, "problem": [7, 8], "come": [7, 8], "type": [7, 12, 13], "under": [7, 8], "hood": 7, "substanti": 7, "amount": 7, "lead": 7, "signific": 7, "space": [7, 11], "save": 7, "simplifi": 7, "involv": [7, 8, 12], "contract": 7, "result": [7, 8, 13], "op": 7, "exploit": 7, "friendli": [7, 8], "still": 7, "convert": [7, 8, 13], "run": [7, 8], "through": 7, "seri": [7, 8, 12], "structur": [7, 8], "wa": 8, "electron": 8, "est": 8, "high": [8, 9], "rank": [8, 13], "part": 8, "boil": 8, "down": 8, "prepar": 8, "initi": 8, "consum": 8, "place": [8, 12], "burden": 8, "solv": 8, "rais": 8, "To": [8, 13], "end": 8, "envis": [8, 12], "sever": 8, "layer": 8, "shown": 8, "box": 8, "label": 8, "At": [8, 13], "top": 8, "write": 8, "second": [8, 13], "quantiz": 8, "form": [8, 13], "enough": 8, "disciplin": 8, "regardless": 8, "output": 8, "strict": 8, "rel": 8, "static": 8, "prefer": 8, "onc": 8, "sourc": 8, "bottom": 8, "titl": 8, "suggest": 8, "content": 8, "execut": 8, "happen": 8, "size": 8, "avail": 8, "resid": 8, "serv": 8, "buffer": [8, 13], "manner": [8, 13], "remain": 8, "knowledg": [8, 13], "asid": 8, "establish": 8, "kernel": [8, 13], "histor": 8, "been": 8, "develop": [8, 10], "tri": 8, "directli": 8, "approach": 8, "first": [8, 13], "represent": 8, "ir": 8, "meant": 8, "admittedli": 8, "cost": 8, "portabl": 8, "direct": 8, "graph": 8, "edg": 8, "node": 8, "reorder": 8, "etc": [8, 12, 15], "task": 8, "job": 8, "mapper": 8, "pool": 8, "along": [8, 15], "input": 8, "executor": 8, "level": 9, "architectur": 9, "terminologi": 10, "sublibrari": 10, "sparsemap": [10, 13, 14], "background": [10, 14], "sparsifi": [10, 14], "c": 10, "wai": [11, 13], "l": 11, "mathbb": 11, "rightarrow": 11, "denot": 11, "member": 11, "subspac": 11, "let": 11, "_": 11, "u_v": 11, "storag": 12, "concern": 12, "rang": 12, "union": 12, "intersect": 12, "domaintrait": 12, "domainpimpl": 12, "domainbas": 12, "charg": 12, "model": 12, "concept": 12, "set": [12, 13], "hold": 12, "singl": 12, "chang": 12, "pimpl": 12, "later": [12, 13], "semant": 12, "separ": 12, "refactor": 12, "itself": 12, "templat": 12, "either": 12, "special": [12, 13], "normal": 12, "redefin": 12, "instead": 12, "mirror": 12, "Of": 12, "note": [12, 14], "four": 12, "elementindex": [12, 13], "tileindex": 12, "distarrai": 13, "copi": 13, "best": 13, "worst": 13, "my": 13, "alia": 13, "main": 13, "instanc": 13, "afford": 13, "flexibl": 13, "ad": 13, "sm": 13, "goal": 13, "tot": 13, "outer": 13, "ind_rank": 13, "inner": 13, "dep_rank": 13, "start": 13, "convers": 13, "There": 13, "now": 13, "scenario": 13, "equal": 13, "former": 13, "valid": 13, "concret": 13, "retriev": 13, "oppos": 13, "inject": 13, "insert": 13, "unifi": 13, "realiz": 13, "long": 13, "cover": 13, "gut": 13, "from_sparse_tensor": 13, "lambda": 13, "pass": 13, "ta": 13, "make_arrai": 13, "std": 13, "ind2mod": 13, "over": 13, "oeidx": 13, "associ": 13, "oedix": 13, "empti": 13, "move": 13, "otherwis": 13, "alloc": 13, "injected_d": 13, "tdomain": 13, "itidx": 13, "ieidx": 13, "add": 13, "kei": 14, "hierarchi": 14, "assumpt": 14, "make_tot_tile_": 14, "length": 15, "colloqui": 15}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"tensorwrapp": [0, 4, 10, 15], "consider": [0, 2, 5, 6, 7, 12], "perform": 0, "user": 0, "experi": 0, "creat": 1, "tensor": [1, 2, 4, 5, 6, 7, 8, 13, 15], "specifi": 1, "detail": 1, "construct": 1, "distribut": 2, "design": [2, 3, 5, 6, 7, 12], "what": [2, 5, 6, 7], "i": [2, 5, 6, 7], "why": [2, 4, 5, 6, 7], "do": [2, 5, 6, 7], "we": [2, 5, 6, 7], "need": [2, 5, 6, 7], "document": [3, 9], "high": 3, "level": 3, "architectur": 3, "class": [3, 12], "motiv": 4, "ar": 4, "suffici": 4, "dsl": 4, "relationship": 5, "shape": 6, "": 6, "sparsiti": 7, "propos": 8, "stack": 8, "develop": 9, "content": [9, 10, 14], "sparsemap": [11, 12], "background": 11, "spars": [12, 14], "map": [12, 14], "librari": 12, "kei": 12, "domain": 12, "hierarchi": 12, "sparsifi": 13, "thing": 13, "note": 13, "assumpt": 13, "make_tot_tile_": 13, "algorithm": 13, "sublibrari": 14, "terminologi": 15, "extent": 15, "mode": 15}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinxcontrib.bibtex": 9, "sphinx": 57}, "alltitles": {"TensorWrapper Considerations": [[0, "tensorwrapper-considerations"]], "Performance": [[0, "performance"]], "User Experience": [[0, "user-experience"]], "Creating a Tensor": [[1, "creating-a-tensor"]], "Specifying Tensor Details": [[1, "specifying-tensor-details"]], "Tensor construction": [[1, "tensor-construction"]], "Distribution Design": [[2, "distribution-design"]], "What is a (tensor) distribution?": [[2, "what-is-a-tensor-distribution"]], "Why do we need a (tensor) distribution?": [[2, "why-do-we-need-a-tensor-distribution"]], "Tensor distribution considerations": [[2, "tensor-distribution-considerations"]], "Design Documentation": [[3, "design-documentation"]], "High-Level Architecture": [[3, "high-level-architecture"]], "Class Design": [[3, "class-design"]], "Motivating TensorWrapper": [[4, "motivating-tensorwrapper"]], "Why Tensors?": [[4, "why-tensors"]], "Are tensors a sufficient DSL?": [[4, "are-tensors-a-sufficient-dsl"]], "Why TensorWrapper?": [[4, "why-tensorwrapper"]], "Tensor Relationships Design": [[5, "tensor-relationships-design"]], "What is a tensor relationship?": [[5, "what-is-a-tensor-relationship"]], "Why do we need (tensor) relationships?": [[5, "why-do-we-need-tensor-relationships"]], "(Tensor) relationship considerations": [[5, "tensor-relationship-considerations"]], "Tensor Shape Design": [[6, "tensor-shape-design"]], "What is a tensor\u2019s shape?": [[6, "what-is-a-tensor-s-shape"]], "Why do we need a tensor\u2019s shape?": [[6, "why-do-we-need-a-tensor-s-shape"]], "Shape Considerations": [[6, "shape-considerations"]], "Tensor Sparsity Design": [[7, "tensor-sparsity-design"]], "What is tensor sparsity?": [[7, "what-is-tensor-sparsity"]], "Why do we need tensor sparsity?": [[7, "why-do-we-need-tensor-sparsity"]], "Sparsity considerations": [[7, "sparsity-considerations"]], "Proposed Tensor Stack": [[8, "proposed-tensor-stack"]], "Developer Documentation": [[9, "developer-documentation"]], "Contents:": [[9, null], [10, null]], "TensorWrapper": [[10, "tensorwrapper"]], "SparseMap Background": [[11, "sparsemap-background"]], "Sparse Map Library Design": [[12, "sparse-map-library-design"]], "Key Considerations": [[12, "key-considerations"]], "Domain Class Hierarchy": [[12, "domain-class-hierarchy"]], "SparseMap Class Hierarchy": [[12, "sparsemap-class-hierarchy"]], "Sparsifying a Tensor": [[13, "sparsifying-a-tensor"]], "Things to note": [[13, "things-to-note"]], "Assumptions": [[13, "assumptions"]], "make_tot_tile_ Algorithm": [[13, "make-tot-tile-algorithm"]], "Sparse Maps Sublibrary": [[14, "sparse-maps-sublibrary"]], "Contents": [[14, null]], "TensorWrapper Terminology": [[15, "tensorwrapper-terminology"]], "Tensor Terminology": [[15, "tensor-terminology"]], "Extent": [[15, "extent"]], "Mode": [[15, "mode"]]}, "indexentries": {}})