Search.setIndex({"docnames": ["index", "sparse_maps/background", "sparse_maps/design", "sparse_maps/from_sparse_map", "sparse_maps/index"], "filenames": ["index.rst", "sparse_maps/background.rst", "sparse_maps/design.rst", "sparse_maps/from_sparse_map.rst", "sparse_maps/index.rst"], "titles": ["TensorWrapper", "SparseMap Background", "Sparse Map Library Design", "Sparsifying a Tensor", "Sparse Maps Sublibrary"], "terms": {"spars": [0, 1], "map": [0, 1, 3], "sublibrari": 0, "sparsemap": [0, 3, 4], "background": [0, 4], "librari": [0, 4], "design": [0, 4], "sparsifi": [0, 4], "tensor": [0, 1, 4], "c": 0, "api": [0, 2], "ar": [1, 2, 3], "wai": [1, 3], "store": [1, 2], "sparsiti": [1, 2], "physic": 1, "speak": 1, "we": [1, 2, 3], "us": [1, 2, 3], "notat": 1, "l": 1, "mathbb": 1, "v": 1, "rightarrow": 1, "u": 1, "denot": 1, "which": [1, 2, 3], "from": [1, 2, 3], "vector": 1, "space": 1, "In": [1, 3], "practic": 1, "each": [1, 2], "member": 1, "subspac": 1, "term": 1, "its": 1, "domain": [1, 3, 4], "let": 1, "_": 1, "A": 1, "u_v": 1, "finish": 1, "thi": [1, 2, 3], "section": 1, "up": [1, 3], "tile": [2, 3], "vs": 2, "element": [2, 3], "storag": 2, "concern": 2, "e": [2, 3], "g": 2, "rang": 2, "etc": 2, "encapsul": 2, "basic": 2, "oper": 2, "union": 2, "intersect": 2, "The": [2, 3], "seri": 2, "involv": 2, "domaintrait": 2, "domainpimpl": 2, "domainbas": 2, "charg": 2, "model": 2, "concept": 2, "i": [2, 3], "set": [2, 3], "like": 2, "object": [2, 3], "hold": 2, "indic": [2, 3], "type": [2, 3], "all": [2, 3], "By": 2, "have": [2, 3], "singl": 2, "avoid": 2, "need": 2, "chang": 2, "multipl": 2, "place": 2, "actual": 2, "data": [2, 3], "It": 2, "envis": 2, "addit": 2, "pimpl": 2, "implement": 2, "later": [2, 3], "point": [2, 3], "differ": 2, "memori": 2, "semant": 2, "implicitli": 2, "separ": 2, "how": 2, "can": 2, "refactor": 2, "without": 2, "itself": 2, "templat": 2, "either": 2, "special": [2, 3], "exist": [2, 3], "index": [2, 3], "normal": 2, "would": 2, "requir": [2, 3], "redefin": 2, "common": 2, "function": [2, 3], "instead": 2, "factor": 2, "out": [2, 3], "mirror": 2, "same": [2, 3], "reason": [2, 3], "Of": 2, "note": [2, 4], "four": 2, "elementindex": [2, 3], "tileindex": 2, "page": 3, "detail": 3, "creat": 3, "an": 3, "t": 3, "distarrai": 3, "sinc": 3, "go": 3, "copi": 3, "At": 3, "best": 3, "abl": 3, "full": 3, "worst": 3, "piec": 3, "To": 3, "my": 3, "knowledg": 3, "alia": 3, "must": 3, "occur": 3, "For": 3, "main": 3, "kernel": 3, "assum": 3, "instanc": 3, "provid": 3, "afford": 3, "most": 3, "flexibl": 3, "defin": 3, "mai": 3, "ad": 3, "other": 3, "given": 3, "sm": 3, "our": 3, "present": 3, "goal": 3, "tot": 3, "outer": 3, "rank": 3, "ind_rank": 3, "inner": 3, "dep_rank": 3, "start": 3, "convers": 3, "class": [3, 4], "There": 3, "now": 3, "two": 3, "scenario": 3, "equal": 3, "less": 3, "than": 3, "former": 3, "depend": 3, "valid": 3, "more": 3, "concret": 3, "0": 3, "1": 3, "take": 3, "mean": 3, "user": 3, "want": 3, "retriev": 3, "oppos": 3, "sai": 3, "offset": 3, "permut": 3, "convert": 3, "second": 3, "inject": 3, "one": 3, "independ": 3, "mode": 3, "get": 3, "word": 3, "reduc": 3, "slice": 3, "form": 3, "insert": 3, "These": 3, "unifi": 3, "realiz": 3, "first": 3, "result": 3, "thu": 3, "long": 3, "code": 3, "manner": 3, "work": 3, "cover": 3, "both": 3, "gut": 3, "from_sparse_tensor": 3, "contain": 3, "lambda": 3, "pass": 3, "ta": 3, "make_arrai": 3, "fill": 3, "std": 3, "ind2mod": 3, "th": 3, "loop": 3, "over": 3, "oeidx": 3, "If": 3, "associ": 3, "oedix": 3, "empti": 3, "move": 3, "otherwis": 3, "alloc": 3, "buffer": 3, "injected_d": 3, "tdomain": 3, "itidx": 3, "ieidx": 3, "add": 3, "return": 3, "kei": 4, "consider": 4, "hierarchi": 4, "thing": 4, "assumpt": 4, "make_tot_tile_": 4, "algorithm": 4}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"tensorwrapp": 0, "content": [0, 4], "sparsemap": [1, 2], "background": 1, "todo": 1, "spars": [2, 4], "map": [2, 4], "librari": 2, "design": 2, "kei": 2, "consider": 2, "domain": 2, "class": 2, "hierarchi": 2, "sparsifi": 3, "tensor": 3, "thing": 3, "note": 3, "assumpt": 3, "make_tot_tile_": 3, "algorithm": 3, "sublibrari": 4}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "sphinx": 56}})