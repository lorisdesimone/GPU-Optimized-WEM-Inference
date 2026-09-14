import warnings
import time  
from .backend import xp
from .fuzzysets import triang_fs
from .metrics import mixed_metric
from .domainDivision import density_div, linear_div, unique_div

# --- Class
class DSregressor:
    
    def __init__(self,
                 fire_of_train_rules=True,
                 fuzzy_output=True,
                 default_division="D",
                 default_n_of_sets=5,
                 min_number_of_sets=2,
                ):
        """
        Initialize the De Simone fuzzy inference system for regression task.
        
        Parameters
        ----------
        fire_of_train_rules : bool, optional
            If True, the firing degree of each rule computed during training
            is incorporated into the defuzzification formula during inference
            This mechanism weights rule activation onnew data by the historical
            reliability of the rule.
    
        fuzzy_output : bool, optional
            If True, conflicting rules sharing identical antecedents are
            resolved using the empowered fuzzy fusion mechanism, where 
            consequents are merged through a weighted average and firing
            degrees through probabilistic OR conorm.  
            If False, conflicts are resolved by selecting the rule with the
            highest firing degree (classical Wang & Mendel behavior).
    
        default_division : str or list, optional
            Either a single letter ('D', 'L', 'U') broadcasted to all features,
            or a list specifying the division type per feature. 
            Elements may be:
                - 'D' : density-based division
                - 'L' : linear division
                - 'U' : unique-value division
                - numeric lists : strictly increasing unique values defining custom pivots
    
        default_n_of_sets : int, optional
            Initial number of fuzzy sets per feature before cross‑validation.
            This value is used as the starting point for the coordinate‑wise
            optimization procedure described in the paper.
    
        min_number_of_sets : int, optional
            Minimum allowed number of fuzzy sets per feature during
            cross‑validation. Prevents overly coarse partitions and ensures
            a minimal expressive capacity of the fuzzy rule base.
        """

        self.fire_of_train_rules = fire_of_train_rules 
        self.fuzzy_output = fuzzy_output
        self.default_division = default_division
        self.default_n_of_sets = default_n_of_sets
        self.min_number_of_sets = min_number_of_sets
        
    def _build_fuzzy_tensor(self, X, build_pivots):
        """
        Compute the fuzzy membership tensor for the dataset X.
    
        If build_pivots is True (used during fit), the pivot points for each
        feature are computed from X and stored in self.pivots. If False
        (used during predict), the stored pivots are reused to ensure
        consistency with the fuzzy sets defined during training.
    
        Parameters
        ----------
        X : array-like of shape (N, R)
            Input dataset.
        build_pivots : bool
            If True, compute and store pivots for each feature.
            If False, reuse previously stored pivots.
    
        Returns
        -------
        ndarray of shape (R, N, P_max)
            Fuzzy membership tensor, where P_max is the maximum number of
            fuzzy sets across features.
        """
        
        # Setting dataframe to correct backend
        X = xp.asarray(X)
        # Metadata
        N, R = X.shape
        # Max number of fuzzy sets
        P_max = int(xp.max(self.n_of_sets))
        # init fuzzy tensor
        F = xp.zeros((R, N, P_max))
        # Cicle on features (could include output feature)

        if build_pivots:
            self.pivots = []   # salva i pivots del train
        
        for r in range(R):
            arr = X[:, r]

            if build_pivots:
                # Pivots are built from X dataframe
                pivots = self.domains_functions[r](arr, int(self.n_of_sets[r]))
                self.pivots.append(pivots)
            else:
                # Pivots are retrived from object memory
                pivots = self.pivots[r]
            
            # Cicle on centers
            for s in range(len(pivots) - 2):
                a = pivots[s]
                b = pivots[s+1]
                c = pivots[s+2]
                # Fuzzy sets infertion of values
                F[r, :, s] = triang_fs(arr, a, b, c)
        
        return F

    def _process_rules(self, mat):
        """
        Remove duplicate rules and resolve conflicts among rules
        sharing the same antecedents.
    
        Parameters
        ----------
        mat : array-like of shape (N, M+2)
            Raw rule matrix: M antecedents, 1 consequent, 1 firing degree.
    
        Returns
        -------
        ndarray of shape (P, M+2)
            Final rule matrix after duplicate removal and conflict resolution.
        """
        
        # Metadata
        n_features = mat.shape[1] - 2
        CONSEQ_COL = n_features
        FIRE_COL = n_features + 1
        
        # --- Duplication elimination
        sort_matrix = mat.T[::-1].copy() # Matrix for lexsort
        # Fire column is inverted because lexsort is ascending, we want the opposite
        sort_matrix[0] = -sort_matrix[0] 
        # Indexes of sorting
        idx = xp.lexsort(sort_matrix)
        # Sorting
        mat_sorted = mat[idx]
    
        # Searching for no duplication rules setting to False
        idx_nodup = xp.any(mat_sorted[1:, :FIRE_COL] != mat_sorted[:-1, :FIRE_COL], axis=1)
        keep = xp.concatenate((xp.array([True], dtype=bool), idx_nodup)) # Dimension correction 
        # Matrix with no duplication
        mat_nodup = mat_sorted[keep]
    
        # --- Conflict resolution
        antecedents = mat_nodup[:,:CONSEQ_COL]
        sort_matrix2 = antecedents.T[::-1] # First key will be last feature
        idx2 = xp.lexsort(sort_matrix2) # Sorting over antecedents
        
        mat2 = mat_nodup[idx2]
        ant2 = antecedents[idx2]
    
        # Wheer antecedents changes we switch groups
        diff2 = xp.any(ant2[1:] != ant2[:-1], axis=1)
    
        group_start = xp.concatenate((xp.array([0]), xp.where(diff2)[0] + 1))
        group_end = xp.concatenate((xp.where(diff2)[0] + 1, xp.array([len(mat2)])))
    
        final_rules = []
    
        # loop over groups
        for s, e in zip(group_start, group_end):
            group = mat2[s:e]
    
            if len(group) == 1:
                final_rules.append(group[0])
                continue
    
            fires = group[:, FIRE_COL]
            consequents = group[:, CONSEQ_COL]
    
            if self.fuzzy_output:
                # --- Fuzzy fusion (De Simone 2026)
                c_new = xp.sum(consequents * fires) / xp.sum(fires)
                fire_new = 1 - xp.prod(1 - fires)
    
                new_rule = xp.concatenate([
                    group[0, :n_features],
                    xp.array([c_new, fire_new])
                ])
    
            else:
                # --- Crisp selection (Wang & Mendel 1992)
                idx_best = xp.argmax(fires)
                new_rule = group[idx_best]
    
            final_rules.append(new_rule)

        self.Rules = xp.stack(final_rules)
        return self.Rules

    def _build_binary_tensor(self):
        """
        Construct the binary one‑hot tensor encoding fuzzy set indices for each rule.
    
        Returns
        -------
        ndarray of shape (n_features, P_max, n_rules)
            Binary tensor where entry [f, s, r] is 1 if rule r uses
            fuzzy set s for feature f, otherwise 0.
        """
        
        # Extract matrix containing H rules for N features
        FS = self.Rules[:, :self.n_features]  # (n_rules, n_features)

        # Retriving max number of sets
        P_max = int(xp.max(self.n_of_sets))
        
        # Binary tensor: onbe-hot encoding of dimension R x P_max x H
        B = (FS.T[:, None, :] == xp.arange(P_max)[None, :, None]) * 1
    
        return B

    def _validate_n_of_sets(self, n_of_sets, n_columns):
        """
        Validate and construct the per‑feature number of fuzzy sets.
    
        Parameters
        ----------
        n_of_sets : int or array-like
            User-provided specification of fuzzy sets per feature.
        n_columns : int
            Total number of columns (features + output).
    
        Returns
        -------
        ndarray of shape (n_columns,)
            Validated number of fuzzy sets. For domain types 'U' and 'X',
            the number of sets is inferred from the domain pivots.
        """
        
        dont_need_a_number = ["U", "X"]
    
        if isinstance(n_of_sets, int):
            if n_of_sets < self.min_number_of_sets:
                raise ValueError(
                    f"n_of_sets must be at least greater than {self.min_number_of_sets}"
                )
            self.n_of_sets = xp.full(n_columns, n_of_sets, dtype=int)
    
        elif hasattr(n_of_sets, "__len__"):
            n_of_sets_new = xp.asarray(n_of_sets, dtype=int)
    
            if n_of_sets_new.ndim != 1:
                raise ValueError("n_of_sets must be 1D")
    
            if len(n_of_sets_new) != n_columns:
                raise ValueError("n_of_sets length mismatch")
    
            if xp.min(n_of_sets_new) < self.min_number_of_sets:
                raise ValueError("n_of_sets contains values below minimum")
    
            self.n_of_sets = n_of_sets_new
    
        else:
            raise ValueError("n_of_sets must be int or array-like")
    
        # Override for domains that do not depend on a number
        for i, dt in enumerate(self.domain_type_list):
            if dt in ("U", "X"):
                pivots = self.domains_functions[i](self.df[:, i], None)
                self.n_of_sets[i] = len(pivots) - 2

    def _build_domains_functions(self, domain_div_test, n_columns):
        """
        Validate the domain division specification and map each feature
        to its corresponding domain division function.
    
        Parameters
        ----------
        domain_div_test : str or list
            Either a single letter ('D', 'L', 'U') broadcasted to all features,
            or a list specifying the division type per feature. 
            Elements may be:
                - 'D' : density-based division
                - 'L' : linear division
                - 'U' : unique-value division
                - numeric lists : strictly increasing unique values defining custom pivots
        n_columns : int
            Number of columns (features + output).
    
        Sets
        ----
        self.domains_functions : list of callables
            One pivot-generating function per feature.
        self.domain_type_list : list of str
            Domain type per feature ('D', 'L', 'U', or 'X').
        """
    
        allowed_letters = ["D", "L", "U"]
        domain_type_list = []
        
        # Case 1: single letter -> replicate
        if isinstance(domain_div_test, str):
            if domain_div_test not in allowed_letters:
                raise ValueError(
                    f"Invalid letter '{domain_div_test}'. Allowed letters are {allowed_letters}."
                )
            domain_div_test = [domain_div_test] * n_columns
        
        # Case 2: list or tuple
        elif isinstance(domain_div_test, (list, tuple)):
            if len(domain_div_test) != n_columns:
                raise ValueError(
                    f"domain_div_test must have length {n_columns}, "
                    f"but has length {len(domain_div_test)}."
                )
        
        # Case 3: invalid type
        else:
            raise TypeError("domain_div_test must be either a string or a list.")

        mapped = []
    
        for elem in domain_div_test:
    
            # Case A: letter -> map directly
            if isinstance(elem, str):
                if elem == "D":
                    mapped.append(density_div)
                    domain_type_list.append(elem)
                    
                elif elem == "L":
                    mapped.append(linear_div)
                    domain_type_list.append(elem)
                    
                elif elem == "U":
                    mapped.append(unique_div)
                    domain_type_list.append(elem)
                    
                else:
                    raise ValueError(
                        f"Invalid letter '{elem}'. Allowed letters are {allowed_letters}."
                    )
                continue
    
            # Case B: numeric list -> validate + map
            if isinstance(elem, (list, tuple, xp.ndarray)):
                
                arr = xp.array(elem)
    
                # Strict monotonicity
                if xp.any(arr[1:] <= arr[:-1]):
                    raise ValueError(
                        f"Numeric list {elem} is not strictly monotonic increasing."
                    )
    
                # Uniqueness
                if arr.size != xp.unique(arr).size:
                    raise ValueError(
                        f"Numeric list {elem} contains duplicate values."
                    )
    
                # Map to lambda
                mapped.append(
                    lambda _, __, arr=arr: xp.pad(arr, (1, 1), mode="edge")
                )
                domain_type_list.append("X")
                
                continue
    
            # Case C: invalid type
            raise TypeError(
                f"Invalid element type {type(elem)}. "
                "Only letters or numeric lists are allowed."
            )

        self.domains_functions = mapped
        self.domain_type_list = domain_type_list

    # Public Functions
    def fit(self, 
        train_df,
        domains_functions_list=None,
        n_of_sets=None,
        _skip_validate=False):
        """
        Fit the WEMDS model on the training dataset.
    
        Parameters
        ----------
        train_df : array-like of shape (N, R+1)
            Training dataset with R input features and one output column.
        domains_functions_list : str or list, optional
            Specification of domain division per feature ('D', 'L', 'U', or
            numeric lists). If a single letter is provided, it is broadcast
            to all columns.
        n_of_sets : int or array-like, optional
            User-provided number of fuzzy sets per feature. Validated unless
            _skip_validate is True.
        _skip_validate : bool, optional
            Internal flag used by cross-validation to bypass validation of
            n_of_sets and domain functions.
    
        Sets
        ----
        self.n_features : int
            Number of input features.
        self.n_of_sets : ndarray
            Validated number of fuzzy sets per feature.
        self.domains_functions : list of callables
            Pivot-generating functions per feature.
        self.domain_type_list : list of str
            Domain type per feature.
        self.y_centers : ndarray
            Centers of the output fuzzy sets.
        self.Rules : ndarray
            Final processed fuzzy rule matrix.
        self.pivots : list of ndarrays
            Pivot points for each feature, computed during training and reused
            during prediction to ensure consistent fuzzy sets.
        """
        
        # Default values from object attributes
        if domains_functions_list is None:
            domains_functions_list = self.default_division
    
        if n_of_sets is None:
            n_of_sets = self.default_n_of_sets
            
        # Dataframe conversion to backend
        df = xp.asarray(train_df)
        y = df[:,-1]
        self.df = df

        # Metadata info
        self.n_features = df.shape[1] - 1
        n_columns = df.shape[1]
        
        # --- Skip validation ---
        if _skip_validate:
            # Skip validation, set only number of sets (used for cross-val)
            self.n_of_sets = n_of_sets
        else:
            # Validation steps of fit function parameters
            self._build_domains_functions(domains_functions_list, n_columns)
            self._validate_n_of_sets(n_of_sets, n_columns)

        #Pivots building
        FT = self._build_fuzzy_tensor(train_df, True)  # Fuzzy tensor

        #Y_centers retriving
        self.y_centers = self.pivots[-1][1:-1]
        
        F = xp.max(FT, axis=2).T                       # Fires matrix
        
        fires_row = xp.prod(F, axis=1)
        fires_column = fires_row.reshape(-1,1)         # Column of fires

        I = xp.argmax(FT, axis=2).T                    # Index matrix
        
        K = xp.hstack((I, fires_column))               # Kingdom Matrix
        
        self.Rules = self._process_rules(K)            # Rules Matrix
    
    # Predict
    def predict(self, X_predict):

        if not hasattr(self, "Rules"):
            raise RuntimeError("Model not fitted")
    
        X = xp.asarray(X_predict)
    
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"predict_df must have {self.n_features} columns, got {X.shape[1]}"
            )
    
        FT_new = self._build_fuzzy_tensor(X, False)
    
        max_fs = max(self.n_of_sets)
    
        BT = self._build_binary_tensor()
    
        W = FT_new @ BT
    
        fire_new = xp.prod(W, axis=0)

        if self.fire_of_train_rules:

            fire_new = fire_new * self.Rules[:,-1][None, :]
        
        total_weights = xp.sum(fire_new, axis=1).reshape(-1,1)
    
        y_rules = self.Rules[:, -2]
    
        remapped_val = xp.interp(
            xp.ascontiguousarray(y_rules),
            xp.arange(len(self.y_centers)),
            self.y_centers
        ).reshape(-1,1)

        out = ((fire_new @ remapped_val) / total_weights).ravel()
    
        return out

    def fit_cross_validation(self, 
                             train_df,
                             validation_df, 
                             domains_functions_list=None,
                             n_of_sets=None,
                             test="r2",
                             print_log=True):
        """
        Optimize the number of fuzzy sets per feature using a greedy
        coordinate-wise search over the validation set.
    
        Parameters
        ----------
        train_df : array-like of shape (N, R+1)
            Training dataset.
        validation_df : array-like of shape (M, R+1)
            Validation dataset.
        n_of_sets : int or array-like, optional
            Initial number of fuzzy sets per feature.
        domains_functions_list : str or list, optional
            Domain division specification per feature.
        test : {"r2", "mse"}, optional
            Metric used to evaluate model performance.
        print_log : bool, optional
            Whether to print progress logs.
    
        Returns
        -------
        n_of_sets_opt : ndarray
            Optimized number of fuzzy sets per feature.
        best_score : float
            Best validation score obtained.
        """

        # Time statistics init
        training_time = 0
        train_step_count = 0

        # Time steps update
        def time_update(start, end):
            training_time += end - start
            train_step_count += 1
            
        # Default values from object attributes
        if domains_functions_list is None:
            domains_functions_list = self.default_division
    
        if n_of_sets is None:
            n_of_sets = self.default_n_of_sets
            
        # ---- Metadata
        df_train = xp.asarray(train_df)
        df_val   = xp.asarray(validation_df)
        n_columns = df_train.shape[1]
        
        # --- Evaluation function ---
        def evaluate(n_sets_to_try, 
                    _skip_validate=True,
                    ):

            start = time.perf_counter()
            
            self.fit(df_train,
                    n_of_sets=n_sets_to_try,
                    _skip_validate=_skip_validate
                    )
            
            end = time.perf_counter()
            time_update(start, end)
            
    
            y_pred = self.predict(df_val[:, :-1])
            y_true = df_val[:, -1]
    
            return mixed_metric(y_pred, y_true, metric=test)

        # --- Log of score ---
        def log_score(n_of_sets_new, new_score, ident):
            print("|"+"----"*ident, end="> ")
            print((f"New best score with n_of_sets: {n_of_sets_new} -> Scored: {new_score}"))

        # First fit
        start = time.perf_counter()
        
        self.fit(df_train,
                domains_functions_list=domains_functions_list,
                n_of_sets=n_of_sets,
                )
        
        end = time.perf_counter()
        time_update(start, end)
        
        # Setting first score
        y_pred = self.predict(df_val[:, :-1])
        y_true = df_val[:, -1]
        best_score = mixed_metric(y_pred, y_true, metric=test)

        # Starting log
        if print_log:
            print("|----> Starting Global Optimization ")
            log_score(self.n_of_sets, best_score, 1)

        # --- GLOBAL OPTIMIZATION PHASE ---
        improved_global = True
        
        # Mask out where type don't need to be modified
        mask = xp.asarray([1 if dt in ("L", "D") else 0 for dt in self.domain_type_list])

        first_step_start_time = time.time()
        
        while improved_global:
            
            improved_global = False

            n_current = self.n_of_sets.copy()
            
            # +1
            n_up_all = n_current + mask
            score_up_all = evaluate(n_up_all)

            candidates = [
                            (best_score, n_current),
                            (score_up_all, n_up_all)
                         ]
            
            # -1 su tutte le feature
            n_down_all = n_current - mask
            n_down_all = xp.maximum(n_down_all, self.min_number_of_sets)
            
            if xp.any(n_down_all < n_current):
                score_down_all = evaluate(n_down_all)
                candidates.append((score_down_all, n_down_all))
        
            # Scegli il migliore
            if test == "r2":
                new_score, new_n = max(candidates, key=lambda x: x[0])
            else:
                new_score, new_n = min(candidates, key=lambda x: x[0])
        
            if new_score != best_score:
                best_score = new_score
                self.n_of_sets = new_n
                improved_global = True
                if print_log:
                    log_score(new_n, best_score, 1)
            else:
                self.n_of_sets = n_current
            
        # --- Coordinate-wise optimizzation ---
        improved = True

        if print_log:
            print("|--------> Starting Coordinate-wise Optimization ")

        second_step_start_time = time.time()
        
        while improved:
            
            improved = False
        
            for i in range(n_columns):
        
                # Skip domains that do NOT depend on n_of_sets
                if self.domain_type_list[i] in ("U", "X"):
                    continue

                n_current = self.n_of_sets.copy()
                n_i = self.n_of_sets[i]
        
                # --- +1 Test
                n_up = n_current.copy()
                n_up[i] = n_i + 1
                score_up = evaluate(n_up)
                
                candidates = [(best_score, self.n_of_sets), (score_up, n_up)]
        
                # --- -1 Test
                if n_i > self.min_number_of_sets:
                    n_down = n_current.copy()
                    n_down[i] = n_i - 1
                    score_down = evaluate(n_down)
                    candidates.append((score_down, n_down))
        
                # --- Scegli il migliore
                if test == "r2": 
                    new_score, new_n = max(candidates, key=lambda x: x[0])
                else:
                    new_score, new_n = min(candidates, key=lambda x: x[0])
        
                if new_score != best_score:
                    best_score = new_score
                    self.n_of_sets = new_n
                    improved = True
                    if print_log:
                        log_score(new_n, new_score,2)
                else:
                    self.n_of_sets = n_current

        # Final fit
        self.fit(df_train,
                 n_of_sets=self.n_of_sets,
                 _skip_validate=True)

        global_ending_time = time.time()

        # Time statistics
        first_step_time = second_step_start_time - first_step_start_time
        second_step_time = global_ending_time - second_step_start_time
        global_optimization_time = first_step_time + second_step_time
        mean_training_time = training_time/train_step_count
        
        print(
                "|" + "----"*3 + "> Cross-validation ended succesfully:" +
                "\n|" + "----"*3 + f"> {self.n_of_sets} -> Best number of set per feature" +
                "\n|" + "----"*3 + f"> {best_score} -> Best score: {test} metric multiplied by coverage" +
                "\n|" + "----"*3 + f"> {global_optimization_time}s -> Optimization total time, of wich" +
                "\n|" + "----"*3 + f"> {first_step_time}s -> First step and {second_step_time}s -> Second step " +
                "\n|" + "----"*3 + f"> {mean_training_time}s -> Mean training time for different configuration. " +
                "\n"
            )

        return 







        