// Complete Learning Plan with Resources and Detailed Tasks
const learningPlan = {
    tonight: {
        title: "🌟 Start Tonight! First Steps",
        description: "Get your environment set up and complete these 4 simple tasks to begin your journey RIGHT NOW!",
        tasks: [
            "⏱️ 30 min: Watch 'PyTorch in 100 seconds' by Fireship + 'PyTorch Tutorial for Beginners' by freeCodeCamp (first 30 min) | Links: https://youtu.be/ORMx45xqWkA + https://youtu.be/V_xro1bcAuA",
            "⏱️ 30 min: Install Anaconda from anaconda.com, then run 'conda install pytorch' in terminal, create your first tensor | Tutorial: https://pytorch.org/get-started/locally/",
            "⏱️ 10 min: Create learning journal - use Notion (notion.so), Google Docs, or simple text file. Title: 'Active Inference Journey - Week 1'",
            "⏱️ 5 min: Bookmark these sites: pytorch.org/docs, github.com/infer-actively/pymdp, activeinference.org, and THIS TRACKER!"
        ]
    },
    
    phase1: {
        title: "Phase 1: Foundations",
        dates: "Oct 28, 2025 - Jan 19, 2026",
        duration: "12 weeks | 14 hours/week | 2 hours/day",
        weeks: [
            {
                title: "Week 1: Python Basics",
                dates: "Oct 28 - Nov 3, 2025",
                days: [
                    {
                        day: 1,
                        task: "Install Python/Anaconda + VS Code. Follow: https://code.visualstudio.com/docs/python/python-tutorial | Create 'hello_world.py' and run it | Read Python Crash Course Ch. 1"
                    },
                    {
                        day: 2,
                        task: "Variables & data types: Watch 'Python for Beginners' by Programming with Mosh (30 min) https://youtu.be/_uQrJ0TkZlc | Practice: Create variables of each type (int, float, str, bool)"
                    },
                    {
                        day: 3,
                        task: "Lists, tuples, dictionaries: Read Python Crash Course Ch. 3-4 | Practice on replit.com: Create a dictionary of your favorite movies with ratings"
                    },
                    {
                        day: 4,
                        task: "Control flow (if/else, loops): Complete exercises at https://www.learnpython.org/en/Conditions + Loops section | Write program: FizzBuzz"
                    },
                    {
                        day: 5,
                        task: "Functions: Watch Corey Schafer's 'Python Functions' https://youtu.be/9Os0o3wzS_I | Create 3 functions: calculate_mean(), find_max(), is_prime()"
                    },
                    {
                        day: 6,
                        task: "Practice day: Solve 5 problems on HackerRank Python basics https://www.hackerrank.com/domains/python | Focus on: Say Hello World, Simple Array Sum, loops"
                    },
                    {
                        day: 7,
                        task: "Weekly project: Build a simple calculator (add, subtract, multiply, divide) using functions | Write in journal: What did I struggle with? What clicked?"
                    }
                ]
            },
            {
                title: "Week 2: NumPy + Math Refresh",
                dates: "Nov 4 - Nov 10, 2025",
                days: [
                    {
                        day: 1,
                        task: "NumPy basics: Watch 'NumPy Tutorial' by Keith Galli https://youtu.be/GB9ByFAIAH4 (first 45 min) | Install: pip install numpy | Create arrays, check shape, dtype"
                    },
                    {
                        day: 2,
                        task: "Array operations: Read https://numpy.org/doc/stable/user/quickstart.html | Practice: indexing, slicing, reshaping | Do: 10 exercises from numpy-100 on GitHub"
                    },
                    {
                        day: 3,
                        task: "Matrix operations: Khan Academy Linear Algebra - Matrices intro https://www.khanacademy.org/math/linear-algebra | Implement: matrix addition, multiplication in NumPy"
                    },
                    {
                        day: 4,
                        task: "Vectors & dot products: Watch 3Blue1Brown 'Essence of Linear Algebra' Ch. 1-3 https://youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab | Code: dot product calculator"
                    },
                    {
                        day: 5,
                        task: "Probability basics: Khan Academy Probability https://www.khanacademy.org/math/statistics-probability | Focus: basic probability, conditional probability, Bayes' theorem"
                    },
                    {
                        day: 6,
                        task: "Practice: Implement mean, standard deviation, covariance in NumPy | Compare with np.mean(), np.std(), np.cov() | Read: https://numpy.org/doc/stable/reference/routines.statistics.html"
                    },
                    {
                        day: 7,
                        task: "Weekly project: Create matrix calculator (transpose, inverse, determinant, eigenvalues) | Use np.linalg | Document what each operation means"
                    }
                ]
            },
            {
                title: "Week 3: Data Handling",
                dates: "Nov 11 - Nov 17, 2025",
                days: [
                    {
                        day: 1,
                        task: "Pandas basics: Watch 'Pandas Tutorial' by Keith Galli https://youtu.be/vmEHCJofslg (1 hour) | Install: pip install pandas | Create DataFrame, explore head(), tail(), info()"
                    },
                    {
                        day: 2,
                        task: "Data manipulation: Read https://pandas.pydata.org/docs/user_guide/10min.html | Practice: read_csv(), filtering, groupby(), merge() | Download sample dataset from Kaggle"
                    },
                    {
                        day: 3,
                        task: "Matplotlib basics: Watch 'Matplotlib Tutorial' by Corey Schafer https://youtu.be/UO98lJQ3QGI | Install: pip install matplotlib | Create: line plot, scatter plot, bar chart"
                    },
                    {
                        day: 4,
                        task: "Seaborn for stats plots: Watch sentdex's Seaborn tutorial https://youtu.be/6GUZXDef2U0 | Install: pip install seaborn | Create: distribution plot, heatmap, pair plot"
                    },
                    {
                        day: 5,
                        task: "Practice project: Download driving dataset from https://github.com/commaai/comma2k19 (small sample) | Load, clean, plot speed over time, acceleration histogram"
                    },
                    {
                        day: 6,
                        task: "Probability distributions: scipy.stats tutorial https://docs.scipy.org/doc/scipy/tutorial/stats.html | Install: pip install scipy | Plot: normal, exponential, Poisson distributions"
                    },
                    {
                        day: 7,
                        task: "Weekly project: Complete data analysis pipeline - load dataset, clean, visualize, calculate statistics, create 5 different plots, write summary in journal"
                    }
                ]
            },
            {
                title: "Week 4: Math Deep Dive",
                dates: "Nov 18 - Nov 24, 2025",
                days: [
                    {
                        day: 1,
                        task: "Calculus refresher: Khan Academy Derivatives https://www.khanacademy.org/math/calculus-1/cs1-derivatives-definition-and-basic-rules | Focus: power rule, chain rule, product rule"
                    },
                    {
                        day: 2,
                        task: "Gradients & partial derivatives: Watch 3Blue1Brown 'Essence of Calculus' Ch. 3-5 https://youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr | Code: gradient calculator"
                    },
                    {
                        day: 3,
                        task: "Bayes' theorem deep dive: Watch 3Blue1Brown 'Bayes theorem' https://youtu.be/HZGCoVF3YvM | Read Parr book Ch. 1 pages 1-10 | Solve 3 Bayesian problems"
                    },
                    {
                        day: 4,
                        task: "Conditional probability: Brilliant.org probability course (free trial) | Practice problems: medical testing, false positives | Implement Bayesian update in Python"
                    },
                    {
                        day: 5,
                        task: "Linear algebra deep dive: 3Blue1Brown series Ch. 4-7 | Topics: linear transformations, matrix multiplication, determinants | Code examples for each"
                    },
                    {
                        day: 6,
                        task: "Eigenvalues & eigenvectors: Watch https://youtu.be/PFDu9oVAE-g (3Blue1Brown) | Implement power iteration algorithm | Use np.linalg.eig() and compare"
                    },
                    {
                        day: 7,
                        task: "Weekly review: Complete 10 problems combining calculus + linear algebra + probability | MIT OCW 18.06 problem set 1 | Write math summary in journal"
                    }
                ]
            },
            {
                title: "Week 5: PyTorch Basics",
                dates: "Nov 25 - Dec 1, 2025",
                days: [
                    {
                        day: 1,
                        task: "Install PyTorch: https://pytorch.org/get-started/locally/ | Watch 'PyTorch Tutorial' by Patrick Loeber https://youtu.be/c36lUUr864M (1 hour) | Create first tensor, check device"
                    },
                    {
                        day: 2,
                        task: "Tensor operations: PyTorch docs https://pytorch.org/docs/stable/tensors.html | Practice: create, reshape, index, slice tensors | Convert between NumPy and PyTorch"
                    },
                    {
                        day: 3,
                        task: "Autograd (automatic differentiation): Watch https://youtu.be/MswxJw-8PvE | Tutorial: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html | Create tensors with requires_grad=True"
                    },
                    {
                        day: 4,
                        task: "Build simple neural network: Follow https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html | Create nn.Module class, forward() method | Print architecture"
                    },
                    {
                        day: 5,
                        task: "Loss functions & optimizers: Read https://pytorch.org/docs/stable/nn.html#loss-functions | Explore: MSELoss, CrossEntropyLoss | Try: SGD, Adam optimizers"
                    },
                    {
                        day: 6,
                        task: "Training loop basics: Watch Aladdin Persson's tutorial https://youtu.be/Jy4wM2X21u0 | Write training loop from scratch: forward pass, loss calculation, backward pass, update"
                    },
                    {
                        day: 7,
                        task: "Weekly project: Linear regression in PyTorch - create dataset, define model, train for 100 epochs, plot loss curve | Compare with sklearn LinearRegression"
                    }
                ]
            },
            {
                title: "Week 6: Neural Networks",
                dates: "Dec 2 - Dec 8, 2025",
                days: [
                    {
                        day: 1,
                        task: "Multi-layer perceptrons: Watch 3Blue1Brown 'Neural Networks' Ch. 1 https://youtu.be/aircAruvnKk | Implement 2-layer MLP with ReLU | Visualize activations"
                    },
                    {
                        day: 2,
                        task: "Activation functions: Read https://pytorch.org/docs/stable/nn.html#non-linear-activations | Implement and plot: sigmoid, tanh, ReLU, LeakyReLU, softmax | When to use each?"
                    },
                    {
                        day: 3,
                        task: "Forward & backpropagation: Watch 3Blue1Brown Ch. 3 'Backpropagation' https://youtu.be/Ilg3gGewQ5U | Implement backprop manually for 1 layer, compare with autograd"
                    },
                    {
                        day: 4,
                        task: "Overfitting & regularization: Watch https://youtu.be/6g0t3Phly2M (Stanford CS230) | Implement: L1, L2 regularization, dropout | Compare training vs validation loss"
                    },
                    {
                        day: 5,
                        task: "Data splitting: Create train/val/test split functions | Read about cross-validation | Implement k-fold cross-validation | Use sklearn.model_selection"
                    },
                    {
                        day: 6,
                        task: "Project setup: Download MNIST dataset | Create data loaders with PyTorch DataLoader | Build 3-layer network | Set up training infrastructure"
                    },
                    {
                        day: 7,
                        task: "Weekly project: Train MNIST classifier - achieve >95% accuracy | Plot training curves | Create confusion matrix | Save best model | Write analysis in journal"
                    }
                ]
            },
            {
                title: "Week 7: Probability in PyTorch",
                dates: "Dec 9 - Dec 15, 2025",
                days: [
                    {
                        day: 1,
                        task: "torch.distributions intro: Read docs https://pytorch.org/docs/stable/distributions.html | Watch tutorial https://youtu.be/ZO1_TGRUqpc | Create Normal, Bernoulli distributions"
                    },
                    {
                        day: 2,
                        task: "Categorical distributions: Create categorical distribution | Sample from it | Calculate log_prob | Useful for: discrete action spaces, classification"
                    },
                    {
                        day: 3,
                        task: "Normal (Gaussian) distributions: Create Normal distribution | Understand mean, std parameters | Sample, calculate log_prob | Plot pdf and samples"
                    },
                    {
                        day: 4,
                        task: "Sampling operations: Practice sampling with sample(), rsample() | Understand reparameterization trick | Critical for: variational inference, active inference"
                    },
                    {
                        day: 5,
                        task: "Log probabilities & KL divergence: Implement KL divergence manually | Use torch.distributions.kl_divergence() | Read: https://arxiv.org/abs/1606.05908 (Variational Inference primer)"
                    },
                    {
                        day: 6,
                        task: "Practice: Implement Gaussian Mixture Model | Sample from mixture | Visualize clusters | Fit to 2D dataset"
                    },
                    {
                        day: 7,
                        task: "Weekly project: Build probabilistic classifier using categorical distributions | Predict class probabilities | Calculate uncertainty | Compare with standard classifier"
                    }
                ]
            },
            {
                title: "Week 8: RL Basics",
                dates: "Dec 16 - Dec 22, 2025",
                days: [
                    {
                        day: 1,
                        task: "MDP fundamentals: Watch David Silver RL Course Lecture 1 https://youtu.be/2pWv7GOvuf0 | Read Sutton & Barto Ch. 3 (free PDF) | Understand: states, actions, rewards"
                    },
                    {
                        day: 2,
                        task: "Value functions: Watch David Silver Lecture 2 https://youtu.be/lfHX2hHRMVQ | Understand V(s) and Q(s,a) | Bellman equations explained"
                    },
                    {
                        day: 3,
                        task: "Q-learning intro: Read https://www.freecodecamp.org/news/an-introduction-to-q-learning-reinforcement-learning-14ac0b4493cc/ | Implement tabular Q-learning"
                    },
                    {
                        day: 4,
                        task: "Policy gradients intro: Watch https://youtu.be/XGmd3wcyDg8 (Arxiv Insights) | Understand difference from value-based methods | Read REINFORCE algorithm"
                    },
                    {
                        day: 5,
                        task: "Implement gridworld: Create GridWorld environment with PyTorch | States: positions, Actions: up/down/left/right, Reward: +1 at goal | Visualize with matplotlib"
                    },
                    {
                        day: 6,
                        task: "Q-learning in PyTorch: Implement DQN (simple version) | Train on gridworld | Plot: reward over episodes, Q-values heatmap | Compare with random policy"
                    },
                    {
                        day: 7,
                        task: "PHASE 1 CHECKPOINT: Review all material from weeks 1-8 | List: 10 things you can do now that you couldn't before | What needs more practice? | Update journal"
                    }
                ]
            }
        ]
    },
    
    phase2: {
        title: "Phase 2: Active Inference Deep Dive",
        dates: "Dec 23, 2025 - May 11, 2026",
        duration: "16 weeks | 20 hours/week | ~3 hours/day",
        weeks: [
            {
                title: "Week 9: Introduction to Active Inference",
                dates: "Dec 23 - Dec 29, 2025",
                days: [
                    {
                        day: 1,
                        task: "Read Parr book Chapter 1 (Introduction) | Take detailed notes | Start qualifying exam literature spreadsheet with: Paper/Chapter, Key Concepts, Relevance to Driving | Add Parr Ch. 1"
                    },
                    {
                        day: 2,
                        task: "Read Parr Chapter 2 (Neuroscience background) | Watch Karl Friston 'Active Inference' intro https://youtu.be/NIu_dJGyIQI | Note biological motivation for framework"
                    },
                    {
                        day: 3,
                        task: "Watch Active Inference Institute - 'What is Active Inference?' https://youtu.be/WwPjqAxRcsY | Create concept map: Free Energy, Inference, Action, Precision"
                    },
                    {
                        day: 4,
                        task: "Read Parr Chapter 3 (Bayesian brain hypothesis) | Understand: brain as inference engine, predictive processing | Connect to your prior Bayes theorem knowledge"
                    },
                    {
                        day: 5,
                        task: "Consolidate notes: Create one-page summary 'What is Active Inference?' | Explain to someone (or record yourself) | Add: how it differs from RL"
                    },
                    {
                        day: 6,
                        task: "Watch complete Active Inference Institute introductory course (2 hours) https://activeinference.org/ | Take breaks, make notes | Add key papers to literature tracker"
                    },
                    {
                        day: 7,
                        task: "Create qualifying exam outline v1: I. Introduction, II. Active Inference Theory, III. Applications to Driving, IV. Future Directions | Share with McDonald for feedback"
                    }
                ]
            },
            {
                title: "Week 10: Free Energy Principle",
                dates: "Dec 30, 2025 - Jan 5, 2026",
                days: [
                    {
                        day: 1,
                        task: "Read Parr Chapter 4 (Free energy principle) | Focus on: variational free energy equation F = E_q[ln q(s) - ln p(o,s)] | Work through each term"
                    },
                    {
                        day: 2,
                        task: "Math practice: Derive free energy bound on surprise | Watch Bogacz (2017) tutorial part 1 https://youtu.be/NIu_dJGyIQI | Verify derivations step-by-step"
                    },
                    {
                        day: 3,
                        task: "Read Bogacz (2017) 'Tutorial on Free Energy' paper https://www.sciencedirect.com/science/article/pii/S0022249615000759 | Add to lit review | Understand predictive coding connection"
                    },
                    {
                        day: 4,
                        task: "Implement free energy calculation in Python | Simple example: sensory observation, prior belief | Calculate F before and after updating beliefs | Plot convergence"
                    },
                    {
                        day: 5,
                        task: "Read Parr Chapter 5 (Message passing) | Understand belief propagation | Connect to: graphical models, Bayesian networks you know from stats"
                    },
                    {
                        day: 6,
                        task: "Practice: Calculate free energy for Gaussian distributions | Given: prior N(μ_prior, σ), observation N(μ_obs, σ) | Compute posterior, calculate F at each step"
                    },
                    {
                        day: 7,
                        task: "Weekly reflection: Can you explain free energy to a colleague? Try! Record explanation or write blog post draft | What's still confusing? List 3 things to revisit"
                    }
                ]
            },
            {
                title: "Week 11: Generative Models",
                dates: "Jan 6 - Jan 12, 2026",
                days: [
                    {
                        day: 1,
                        task: "Read Parr Chapter 6 (POMDPs) | Understand Partially Observable Markov Decision Processes | Key insight: agent doesn't observe states directly, only observations"
                    },
                    {
                        day: 2,
                        task: "Implement simple POMDP in Python | Example: tiger problem or gridworld with noisy observations | Define: states, actions, observations, transition probabilities"
                    },
                    {
                        day: 3,
                        task: "A, B, C, D matrices explained in detail | A: likelihood mapping (obs given state), B: transitions, C: preferences, D: initial beliefs | Create visual diagram"
                    },
                    {
                        day: 4,
                        task: "Read pymdp documentation thoroughly https://pymdp-rtd.readthedocs.io/ | Install: pip install inferactively-pymdp | Run getting started examples"
                    },
                    {
                        day: 5,
                        task: "Install and explore pymdp: Follow 'pymdp Fundamentals' tutorial https://pymdp-rtd.readthedocs.io/en/latest/notebooks/pymdp_fundamentals.html | Run all code cells"
                    },
                    {
                        day: 6,
                        task: "Implement T-Maze gridworld with pymdp | States: positions + context, Actions: move, Observations: visual cues | Train agent to find reward"
                    },
                    {
                        day: 7,
                        task: "Weekly project: Build your own simple generative model from scratch | Choose scenario: foraging, navigation, or simple game | Document all matrices A,B,C,D | Visualize behavior"
                    }
                ]
            },
            {
                title: "Week 12: Active Inference for Action",
                dates: "Jan 13 - Jan 19, 2026",
                days: [
                    {
                        day: 1,
                        task: "Read Parr Chapter 7 (Planning as inference) | Understand: action selection through minimizing expected free energy | G = E_q[F] where F is free energy of future"
                    },
                    {
                        day: 2,
                        task: "Expected Free Energy (EFE) explained: G = Ambiguity + Risk | Ambiguity: uncertainty about observations | Risk: distance from preferred outcomes | Work through math"
                    },
                    {
                        day: 3,
                        task: "Implement EFE calculation in Python | Create toy example with 2 policies | Calculate G for each | Choose policy with lowest G | Understand information-seeking behavior"
                    },
                    {
                        day: 4,
                        task: "Read pymdp 'Active Inference from Scratch' tutorial https://pymdp-rtd.readthedocs.io/en/latest/notebooks/active_inference_from_scratch.html | This is gold - study carefully!"
                    },
                    {
                        day: 5,
                        task: "Code along with 'Active Inference from Scratch' | Type every line yourself, don't copy-paste | Run experiments: change parameters, see what happens"
                    },
                    {
                        day: 6,
                        task: "Modify tutorial for different scenario | Change: number of states, observations, or add new action | Debug issues | Get it working and understand why"
                    },
                    {
                        day: 7,
                        task: "CHECKPOINT: Build complete active inference agent in T-Maze | Agent should: explore when uncertain, exploit when confident | Analyze: info-seeking vs reward-seeking behavior | Document findings"
                    }
                ]
            },
            {
                title: "Week 13: McDonald's Papers - Deep Dive 1",
                dates: "Jan 20 - Jan 26, 2026",
                days: [
                    {
                        day: 1,
                        task: "Read Wei et al. (2024) 'Active inference models of AV takeovers' https://journals.sagepub.com/doi/10.1177/00187208241295932 | This is your lab's work! Study methodology carefully"
                    },
                    {
                        day: 2,
                        task: "Deep dive Wei et al. (2024): Annotate paper with questions | How did they model trust? What was their generative model structure? What were the A, B, C matrices?"
                    },
                    {
                        day: 3,
                        task: "Read Wei et al. (2022) 'Modeling driver responses to automation failures' https://ieeexplore.ieee.org/document/9737359 | Compare approach with 2024 paper | What evolved?"
                    },
                    {
                        day: 4,
                        task: "Read Engström et al. (2024) 'Resolving uncertainty on the fly' https://www.frontiersin.org/articles/10.3389/fnbot.2024.1341750 | Focus on: adaptive driving as active inference"
                    },
                    {
                        day: 5,
                        task: "Create detailed method notes on Wei papers: List 1) Experiment design, 2) Data collected, 3) Model architecture, 4) Parameter estimation, 5) Results | Compare papers"
                    },
                    {
                        day: 6,
                        task: "Add all 3 papers to qualifying exam lit review with detailed summaries | Create table comparing approaches | Identify research gaps they mention"
                    },
                    {
                        day: 7,
                        task: "Write synthesis: 'How is active inference applied to driver behavior?' (2 pages) | Include: why it's suitable, what it captures that other models don't, limitations"
                    }
                ]
            },
            {
                title: "Week 14: Driver Behavior Modeling",
                dates: "Jan 27 - Feb 2, 2026",
                days: [
                    {
                        day: 1,
                        task: "Read Parr Chapter 9 (Continuous time active inference) | Understand: extending to continuous states/time | Generalized coordinates of motion | Mountain car example"
                    },
                    {
                        day: 2,
                        task: "Review car-following models: Read about IDM (Intelligent Driver Model) https://en.wikipedia.org/wiki/Intelligent_driver_model | Implement basic IDM in Python"
                    },
                    {
                        day: 3,
                        task: "Study Optimal Velocity Model (OVM) | Read Bando et al. 1995 paper | Implement OVM | Compare IDM vs OVM behavior | How do they differ? Plot trajectories"
                    },
                    {
                        day: 4,
                        task: "Read from proposal: papers on driver behavior models [59,89] | Add to lit review | How does active inference extend traditional car-following models?"
                    },
                    {
                        day: 5,
                        task: "Implement simple car-following in Python | Create: lead vehicle trajectory, following vehicle using IDM | Plot: headway, speed over time | Add stochasticity"
                    },
                    {
                        day: 6,
                        task: "Add active inference to car-following: Model as POMDP | States: relative velocity/distance, Actions: acceleration, Observations: visual perception | Define A,B,C,D matrices"
                    },
                    {
                        day: 7,
                        task: "Weekly project: Simulate driver following behavior with active inference | Compare with IDM baseline | Which is more human-like? Plot both trajectories | Write analysis"
                    }
                ]
            },
            {
                title: "Week 15: Lab Code Integration",
                dates: "Feb 3 - Feb 9, 2026",
                days: [
                    {
                        day: 1,
                        task: "Meeting prep: List questions for McDonald about existing code | Topics: code structure, dependencies, running examples, where to start contributing"
                    },
                    {
                        day: 2,
                        task: "Get existing lab code from McDonald | Clone repository | Read README | Set up environment: install all dependencies | Run 'hello world' example"
                    },
                    {
                        day: 3,
                        task: "Understand code structure: Map out files/folders | What does each module do? | Create diagram of code architecture | Identify: data loading, model definition, training"
                    },
                    {
                        day: 4,
                        task: "Run existing examples: Execute all demo scripts | Compare outputs with paper results | If errors occur: debug with lab mates or McDonald | Document setup process"
                    },
                    {
                        day: 5,
                        task: "Modify code for new scenario: Change 1 parameter (e.g., number of states, observation noise) | Re-run | Compare results | Understand what parameter controls what"
                    },
                    {
                        day: 6,
                        task: "Experiment day: Try different parameters systematically | Create plots showing: how behavior changes with parameter values | Share findings with lab group"
                    },
                    {
                        day: 7,
                        task: "Document what you learned: Write 'Lab Code Guide' for yourself | Include: how to run, what each file does, common errors and fixes | Will help future you!"
                    }
                ]
            },
            {
                title: "Week 16: Trust and Takeover Modeling",
                dates: "Feb 10 - Feb 16, 2026",
                days: [
                    {
                        day: 1,
                        task: "Read evidence accumulation models: Ratcliff & McKoon (2008) diffusion model https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2474742/ | Understand drift-diffusion process"
                    },
                    {
                        day: 2,
                        task: "How does active inference relate to drift-diffusion? | Read: connection between accumulation and inference | Both accumulate evidence over time! Document similarities/differences"
                    },
                    {
                        day: 3,
                        task: "Read Lee & See (2004) 'Trust in automation' https://journals.sagepub.com/doi/10.1518/hfes.46.1.50_30392 | Foundational paper! Add to lit review | How is trust defined?"
                    },
                    {
                        day: 4,
                        task: "Implement trust dynamics in active inference: Trust as prior belief in automation competence | High trust = high C matrix preference for automation | Model erosion over time"
                    },
                    {
                        day: 5,
                        task: "Model driver takeover decision: State: automation engaged/disengaged, Observations: vehicle behavior, Actions: takeover/monitor | Takeover when: prior belief too low"
                    },
                    {
                        day: 6,
                        task: "Add papers to lit review: All trust/takeover papers from proposal [27,60-66,76-78] | Create subsection: 'Trust Modeling in Automated Vehicles' | Synthesize approaches"
                    },
                    {
                        day: 7,
                        task: "MID-POINT CHECK: Present to lab mate | 20 min presentation: Active inference foundations, application to driving, trust modeling | Get feedback | What's unclear? Revise notes"
                    }
                ]
            },
            {
                title: "Week 17: Preference Modeling",
                dates: "Mar 17 - Mar 23, 2026",
                days: [
                    {
                        day: 1,
                        task: "Re-read proposal Thrust 1 in detail | Take notes on: preference elicitation methods, preference dimensionality, preference malleability | How will this be measured?"
                    },
                    {
                        day: 2,
                        task: "How to operationalize preferences in active inference? | C matrix encodes preferences (prior preferences over observations) | Different drivers = different C matrices | Design experiment"
                    },
                    {
                        day: 3,
                        task: "Prior preferences (C matrices) explained deeply | C encodes 'how much do I want to see this observation?' | Example: aggressive driver prefers small headways | Code examples"
                    },
                    {
                        day: 4,
                        task: "Implement preference learning: Given driver behavior data, infer their C matrix | Use inverse active inference | Compare with inverse RL approaches"
                    },
                    {
                        day: 5,
                        task: "Read IRL papers from proposal [41-43] | Inverse Reinforcement Learning for driving | Add to lit review | Compare IRL vs active inference for preference learning | Pros/cons"
                    },
                    {
                        day: 6,
                        task: "Compare active inference to IRL for preferences: Create comparison table | Dimensions: data efficiency, interpretability, uncertainty quantification, computational cost"
                    },
                    {
                        day: 7,
                        task: "Weekly project: Model individual driver preferences | Create 3 synthetic drivers: aggressive, moderate, conservative | Different C matrices | Simulate their behavior | Visualize differences"
                    }
                ]
            },
            {
                title: "Week 18: Hierarchical Models",
                dates: "Mar 24 - Mar 30, 2026",
                days: [
                    {
                        day: 1,
                        task: "Read Parr Chapter 11 (Structure learning and hierarchical models) | Understand: multiple levels of inference | Low level: immediate actions, High level: goals/plans"
                    },
                    {
                        day: 2,
                        task: "Hierarchical active inference explained: Multiple coupled generative models | Higher levels provide context/priors to lower levels | Example: driving to destination (high) vs steering (low)"
                    },
                    {
                        day: 3,
                        task: "Multi-scale modeling: How does individual behavior (micro) scale to traffic flow (macro)? | Read from proposal: papers on traffic flow theory [3-14] | Add to lit review"
                    },
                    {
                        day: 4,
                        task: "Understand emergent traffic phenomena: Stop-and-go waves, capacity drop | How do individual driver decisions aggregate? | Read systems perspective papers"
                    },
                    {
                        day: 5,
                        task: "Read papers on multi-agent systems from proposal [50,51] | How do multiple AVs interact? | Consider: communication, coordination, emergent behavior"
                    },
                    {
                        day: 6,
                        task: "Implement multi-agent simulation: Create 5-car platoon with active inference agents | Each has own generative model | Observe: how does disturbance propagate? String stability?"
                    },
                    {
                        day: 7,
                        task: "Weekly reflection: How does this connect to qualifying exam? | Update outline | Add section: 'Multi-scale modeling: from micro to macro' | What questions remain?"
                    }
                ]
            },
            {
                title: "Week 19: Deep Active Inference",
                dates: "Mar 31 - Apr 6, 2026",
                days: [
                    {
                        day: 1,
                        task: "Read papers on deep active inference | Ueltzhöffer (2018) 'Deep Active Inference' | Millidge et al. (2020) 'Deep Active Inference as Variational Policy Gradients' | Add to lit review"
                    },
                    {
                        day: 2,
                        task: "Combining neural networks with active inference: NNs learn generative model components (A,B matrices) | Benefits: handle high-dim observations (images), scale to complex tasks"
                    },
                    {
                        day: 3,
                        task: "Variational Autoencoder (VAE) review: Watch Arxiv Insights 'VAE' https://youtu.be/9zKuYvjFFS8 | Read Kingma & Welling (2013) | Understand: encoder, decoder, reparameterization"
                    },
                    {
                        day: 4,
                        task: "Implement VAE in PyTorch: Follow tutorial https://github.com/pytorch/examples/tree/main/vae | Train on MNIST | Visualize: latent space, reconstructions, samples from prior"
                    },
                    {
                        day: 5,
                        task: "Connect VAE to active inference: VAE encoder = inference network (approximates posterior), VAE decoder = generative model | Both minimize variational free energy!"
                    },
                    {
                        day: 6,
                        task: "World models: Read Ha & Schmidhuber (2018) 'World Models' https://worldmodels.github.io/ | Learn environment dynamics | Use for planning | Connect to active inference"
                    },
                    {
                        day: 7,
                        task: "Weekly project: Build neural network generative model | Use CNN to process driving scene images | Learn to predict next frame | Use for active inference planning"
                    }
                ]
            },
            {
                title: "Week 20: Literature Synthesis",
                dates: "Apr 7 - Apr 13, 2026",
                days: [
                    {
                        day: 1,
                        task: "Review ALL papers collected so far | Count them | Should have ~40-50 papers in lit review spreadsheet | Read abstracts of any you haven't read yet"
                    },
                    {
                        day: 2,
                        task: "Organize papers by theme: 1) Active inference theory, 2) Driver behavior, 3) Trust/takeover, 4) Traffic flow, 5) Multi-agent systems | Use spreadsheet tabs or tags"
                    },
                    {
                        day: 3,
                        task: "Identify literature gaps: What questions are unanswered? | From proposal: bidirectional alignment, preference shaping, multi-scale modeling | Write gap analysis (1 page)"
                    },
                    {
                        day: 4,
                        task: "Create concept map connecting papers: Use tool like Miro, draw.io, or paper | Show: how different research areas connect | Where does your research fit?"
                    },
                    {
                        day: 5,
                        task: "Write draft outline of qualifying exam: Detailed! | Include: section headings, key papers in each section, figures to create | Aim for 5-7 main sections"
                    },
                    {
                        day: 6,
                        task: "Get feedback from McDonald on outline | Schedule 30min meeting | Come with specific questions | Take notes on his suggestions | Revise outline based on feedback"
                    },
                    {
                        day: 7,
                        task: "CHECKPOINT: You're 50% done with lit review! | Self-assessment: What do you understand well? What needs deeper study? | Plan weeks 21-24 based on this assessment"
                    }
                ]
            },
            {
                title: "Week 21: Reproduce Lab Results Part 1",
                dates: "Apr 14 - Apr 20, 2026",
                days: [
                    {
                        day: 1,
                        task: "Choose one McDonald paper to fully reproduce | Suggestion: Wei et al. (2022) driver response paper | Read methods section in extreme detail | List all data/code needed"
                    },
                    {
                        day: 2,
                        task: "Get data: Check if data is available from paper repository or ask McDonald | Load data | Explore: check distributions, plot time series, understand structure"
                    },
                    {
                        day: 3,
                        task: "Implement model from paper: Translate mathematical description to code | Start with simplest version | Test on toy data first before real data"
                    },
                    {
                        day: 4,
                        task: "Parameter estimation: How did paper estimate parameters? | Implement estimation procedure | If using lab code, understand each step | Run on subset of data first"
                    },
                    {
                        day: 5,
                        task: "Train/run full model: On complete dataset | Compare your results with paper's figures | Do numbers match? If not, why? | Debug discrepancies"
                    },
                    {
                        day: 6,
                        task: "Validate results: Reproduce key figures from paper | Quantitative comparison: MSE, correlations | Qualitative: do patterns look right? | Document any differences"
                    },
                    {
                        day: 7,
                        task: "Consolidate learning: Write detailed documentation of reproduction attempt | What worked? What was challenging? | Create tutorial for lab mates | Share with McDonald"
                    }
                ]
            },
            {
                title: "Week 22: Reproduce Lab Results Part 2",
                dates: "Apr 21 - Apr 27, 2026",
                days: [
                    {
                        day: 1,
                        task: "Deep dive into model details: Study every equation in paper | Derive results yourself | Check: are there typos? Ambiguities? | Ask McDonald for clarification if needed"
                    },
                    {
                        day: 2,
                        task: "Sensitivity analysis: How do results change with parameters? | Vary key parameters ±20% | Plot: parameter vs metric | Which parameters matter most?"
                    },
                    {
                        day: 3,
                        task: "Ablation study: Remove model components one at a time | Example: What if no uncertainty? No preference learning? | Measure: how much does performance drop?"
                    },
                    {
                        day: 4,
                        task: "Alternative approaches: Try different modeling choices | Example: different inference algorithm, prior distributions | Compare: does it improve results? When/why?"
                    },
                    {
                        day: 5,
                        task: "Visualization: Create comprehensive figures | Show: model architecture, data examples, results, comparisons | Use matplotlib/seaborn | Make publication-quality"
                    },
                    {
                        day: 6,
                        task: "Write-up: Create mini-paper (5-10 pages) documenting reproduction | Include: motivation, methods, results, discussion | Use LaTeX or Overleaf | Add to portfolio"
                    },
                    {
                        day: 7,
                        task: "Present to lab: 15-20 min presentation | Show reproduction results | Discuss: insights gained, challenges faced | Get feedback | This is great exam practice!"
                    }
                ]
            },
            {
                title: "Week 23: Nudging & Boosting",
                dates: "Apr 28 - May 4, 2026",
                days: [
                    {
                        day: 1,
                        task: "Read behavioral science papers on nudging: Thaler & Sunstein 'Nudge' [18] | Focus on: choice architecture, defaults | How to guide behavior without forcing?"
                    },
                    {
                        day: 2,
                        task: "Read about boosting: Hertwig & Grüne-Yanoff (2017) [20-22] | Difference from nudging: empowers through information vs subtle manipulation | Ethical considerations"
                    },
                    {
                        day: 3,
                        task: "How to model nudging in active inference? | Nudge = modify choice architecture = change policy selection without changing preferences | Implement: default settings example"
                    },
                    {
                        day: 4,
                        task: "How to model boosting in active inference? | Boost = provide information = reduce uncertainty = change A matrix (improve observation accuracy) | Implement: information feedback"
                    },
                    {
                        day: 5,
                        task: "Read papers on nudging/boosting in driving [54,55,62] from proposal | How effective are these interventions? | What mechanisms work? | Add to lit review"
                    },
                    {
                        day: 6,
                        task: "Implement preference shaping: Create scenario where driver has suboptimal preference | Apply: nudge (default AV setting) or boost (show traffic impact info) | Measure: behavior change"
                    },
                    {
                        day: 7,
                        task: "Weekly project: Compare nudging vs boosting effectiveness | Simulate: multiple drivers with varied preferences | Measure: which intervention leads to better outcomes? | Discuss ethics"
                    }
                ]
            },
            {
                title: "Week 24: Traffic Simulation",
                dates: "May 5 - May 11, 2026",
                days: [
                    {
                        day: 1,
                        task: "Learn SUMO (Simulation of Urban MObility) | Install: https://sumo.dlr.de/docs/Installing.html | Tutorial: https://sumo.dlr.de/docs/Tutorials/index.html | Create first road network"
                    },
                    {
                        day: 2,
                        task: "SUMO basics: Create highway scenario | Add vehicles | Configure: speed, acceleration, deceleration | Run simulation | Visualize with sumo-gui | Export data"
                    },
                    {
                        day: 3,
                        task: "Alternative: ProjectChrono (if used in lab) | Download from https://projectchrono.org/ | Follow getting started guide | Create simple driving scenario"
                    },
                    {
                        day: 4,
                        task: "Python integration with SUMO: Use TraCI (Traffic Control Interface) | Control vehicles from Python | Tutorial: https://sumo.dlr.de/docs/TraCI.html | Test basic commands"
                    },
                    {
                        day: 5,
                        task: "Integrate active inference agents: Replace SUMO's car-following with your active inference model | Read vehicle state, compute action, send back to SUMO | Test with 1 vehicle"
                    },
                    {
                        day: 6,
                        task: "Multi-agent traffic simulation: Add 10 active inference vehicles | Mix with normal SUMO vehicles | Observe: string stability, throughput, stop-and-go patterns | Record metrics"
                    },
                    {
                        day: 7,
                        task: "PHASE 2 COMPLETE: Review weeks 9-24 | Major achievements: understand active inference, apply to driving, reproduce research | Update qualifying exam outline v2 | What remains for Phase 3?"
                    }
                ]
            }
        ]
    },
    
    phase3: {
        title: "Phase 3: Mastery & Exam Prep",
        dates: "May 12 - Aug 31, 2026",
        duration: "16 weeks | 10-14 hours/week",
        weeks: [
            {
                title: "Week 25: Literature Review Writing 1",
                dates: "May 12 - May 18, 2026",
                days: [
                    {
                        day: 1,
                        task: "Finish reading ALL cited papers in proposal | Currently should have ~50 papers | Aim for 60-70 total | Focus on: papers you haven't read yet, fill gaps"
                    },
                    {
                        day: 2,
                        task: "Read all active inference foundational papers | Papers by Karl Friston, Thomas Parr | Create timeline: evolution of active inference field | Add all to lit review"
                    },
                    {
                        day: 3,
                        task: "Read all driver behavior papers thoroughly | Focus on: human factors, trust, takeover behavior | From proposal refs [24,27,31-34,60-66] | Take detailed notes"
                    },
                    {
                        day: 4,
                        task: "Read all traffic flow papers | Capacity drop, stop-and-go traffic, AV impact on traffic [3-15,50-51,82-88] | Understand: macro-level consequences"
                    },
                    {
                        day: 5,
                        task: "Read supplementary papers: Search Google Scholar for 'active inference driving' | Find recent 2024-2025 papers | Add 5-10 more papers to lit review"
                    },
                    {
                        day: 6,
                        task: "Complete literature review spreadsheet | Ensure all 60-70 papers have: full citation, summary, key findings, relevance to your work | Color code by theme"
                    },
                    {
                        day: 7,
                        task: "Weekly goal: Finish reading phase | From now on, focus shifts to writing and synthesizing | Create annotated bibliography document | Share with McDonald for feedback"
                    }
                ]
            },
            {
                title: "Week 26: Literature Review Writing 2",
                dates: "May 19 - May 25, 2026",
                days: [
                    {
                        day: 1,
                        task: "Write Section 1: Active Inference Foundations | Subsections: Free energy principle, Generative models, Planning as inference | Target: 5-7 pages | Use LaTeX or Overleaf"
                    },
                    {
                        day: 2,
                        task: "Continue Section 1: Add figures | Create: generative model diagram, free energy calculation flowchart, example POMDP | Explain each figure in text"
                    },
                    {
                        day: 3,
                        task: "Write Section 1.4: Comparison with other approaches | Active inference vs RL vs optimal control | Table comparing: assumptions, strengths, limitations | When to use which?"
                    },
                    {
                        day: 4,
                        task: "Write Section 2 intro: Active Inference in Robotics/AI | Brief overview: applications beyond driving | Robotics, navigation, multi-agent systems | Sets context"
                    },
                    {
                        day: 5,
                        task: "Edit Section 1: Read aloud, check flow | Are arguments clear? Sufficient detail? | Get feedback from lab mate | Revise based on comments | Aim for clarity"
                    },
                    {
                        day: 6,
                        task: "Create comprehensive figure: Timeline of active inference development | Major papers, key milestones, applications | Use draw.io or PowerPoint | Make it visual and informative"
                    },
                    {
                        day: 7,
                        task: "Weekly review: Section 1 should be ~80% complete | Share draft with McDonald | Schedule meeting to discuss | Take notes on his feedback | Plan revisions"
                    }
                ]
            },
            {
                title: "Week 27: Literature Review Writing 3",
                dates: "May 26 - Jun 1, 2026",
                days: [
                    {
                        day: 1,
                        task: "Write Section 2: Driver Behavior Modeling | Subsections: Traditional approaches (IDM, OVM), Human factors (trust, takeover), Active inference approaches | Target: 7-10 pages"
                    },
                    {
                        day: 2,
                        task: "Section 2.1: Traditional car-following models | Describe: physics-based, data-driven | Equations, strengths, limitations | Why insufficient for human-AV interaction?"
                    },
                    {
                        day: 3,
                        task: "Section 2.2: Human factors in automated driving | Trust dynamics, mode awareness, takeover behavior | Synthesize literature | What factors influence takeover decisions?"
                    },
                    {
                        day: 4,
                        task: "Section 2.3: Active inference for driver modeling | Review McDonald's papers in detail | How does active inference capture: preferences, uncertainty, adaptation?"
                    },
                    {
                        day: 5,
                        task: "Create figures for Section 2: Driver behavior examples, takeover decision process, active inference model architecture | Use data from lab if available"
                    },
                    {
                        day: 6,
                        task: "Write Section 2.4: Preference modeling | How are preferences operationalized? | C matrices, inverse inference | Compare: stated vs revealed preferences | Challenges"
                    },
                    {
                        day: 7,
                        task: "Edit Section 2: Check clarity, flow, completeness | Are all claims cited? | Get feedback from another grad student | Revise | Section 2 should be ~80% done"
                    }
                ]
            },
            {
                title: "Week 28: Literature Review Writing 4",
                dates: "Jun 2 - Jun 8, 2026",
                days: [
                    {
                        day: 1,
                        task: "Write Section 3: Multi-scale Traffic Dynamics | Subsections: Traffic flow theory, Micro-macro connection, AV impact on traffic | Target: 6-8 pages"
                    },
                    {
                        day: 2,
                        task: "Section 3.1: Traffic flow fundamentals | Fundamental diagram, capacity, stability | Stop-and-go waves, capacity drop | Why do these phenomena occur?"
                    },
                    {
                        day: 3,
                        task: "Section 3.2: From individual behavior to traffic flow | How do micro-level decisions aggregate? | String stability analysis | Perturbation amplification"
                    },
                    {
                        day: 4,
                        task: "Section 3.3: AV impact on traffic | Review literature on AV traffic effects | Mixed traffic challenges | Potential benefits and risks | From proposal refs [50-51,82-88]"
                    },
                    {
                        day: 5,
                        task: "Section 3.4: Bidirectional influence | Individual ↔ System interaction | How system affects individual, individual affects system | Novel contribution of your research"
                    },
                    {
                        day: 6,
                        task: "Create traffic simulation figures | If you have simulation results, include | Otherwise: diagram showing micro-macro connection, conceptual figures | Make it visual"
                    },
                    {
                        day: 7,
                        task: "Complete draft of Section 3 | Edit for clarity | Ensure smooth transitions between sections | Literature review should now be 70-80% complete overall | Share with McDonald"
                    }
                ]
            },
            {
                title: "Week 29: Advanced Implementation 1",
                dates: "Jun 9 - Jun 15, 2026",
                days: [
                    {
                        day: 1,
                        task: "Project: Build complete driver-AV interaction model | Combine everything learned | Components: driver generative model, AV controller, trust dynamics | Start with architecture design"
                    },
                    {
                        day: 2,
                        task: "Implement driver model: States (automation engaged, trust level, traffic conditions), Actions (takeover, monitor), Observations (vehicle behavior) | Code A,B,C,D matrices"
                    },
                    {
                        day: 3,
                        task: "Implement AV controller: Simple adaptive cruise control with settings | Make it configurable: aggressive, moderate, conservative styles | Test in isolation first"
                    },
                    {
                        day: 4,
                        task: "Integrate driver + AV: Closed-loop simulation | Driver observes AV behavior, updates trust, decides takeover | AV responds to driver state | Test interaction dynamics"
                    },
                    {
                        day: 5,
                        task: "Add traffic environment: Include lead vehicle with realistic behavior | Scenarios: steady flow, deceleration, stop-and-go | Driver-AV response to different scenarios"
                    },
                    {
                        day: 6,
                        task: "Debug and refine: Fix any instabilities | Tune parameters for realistic behavior | Compare with human data if available | Iterate until behavior looks reasonable"
                    },
                    {
                        day: 7,
                        task: "Create comprehensive test suite: Multiple scenarios, parameter sweeps | Record: takeover frequency, trust dynamics, safety metrics | Save results for analysis next week"
                    }
                ]
            },
            {
                title: "Week 30: Advanced Implementation 2",
                dates: "Jun 16 - Jun 22, 2026",
                days: [
                    {
                        day: 1,
                        task: "Implement preference learning from data | Given: driver behavior trajectories | Infer: their C matrix (preferences) | Use maximum likelihood or Bayesian approach"
                    },
                    {
                        day: 2,
                        task: "Test preference learning: Generate synthetic data from known preferences | Run inference | Check: do recovered preferences match ground truth? | Assess accuracy"
                    },
                    {
                        day: 3,
                        task: "Apply to real/realistic data: If available from lab, use human driver data | Otherwise: create realistic synthetic scenarios | Infer preferences for multiple drivers"
                    },
                    {
                        day: 4,
                        task: "Analyze preference heterogeneity: Cluster drivers by learned preferences | Visualize: preference distributions, driver types | How much variability exists?"
                    },
                    {
                        day: 5,
                        task: "Implement preference personalization: Given learned preferences, customize AV behavior | Test: does personalization reduce takeovers? | Quantify improvement"
                    },
                    {
                        day: 6,
                        task: "Create visualizations: Model behavior plots, trust dynamics, preference learning convergence | Use matplotlib/seaborn | Make publication-quality figures"
                    },
                    {
                        day: 7,
                        task: "Document everything: Write technical documentation | Include: model equations, implementation details, parameter values | This becomes Methods section draft"
                    }
                ]
            },
            {
                title: "Week 31: Advanced Implementation 3",
                dates: "Jun 23 - Jun 29, 2026",
                days: [
                    {
                        day: 1,
                        task: "Parameter sensitivity analysis: Identify key model parameters | Vary each ±50% | Measure: impact on outcomes (takeover rate, trust, safety) | Create sensitivity plots"
                    },
                    {
                        day: 2,
                        task: "Systematic parameter sweep: Use grid search or Sobol sequences | Sample parameter space comprehensively | Record results in database | Might take several hours to run"
                    },
                    {
                        day: 3,
                        task: "Analyze sensitivity results: Which parameters matter most? | Ranking by importance | Implications: which need careful tuning? Which are robust? | Create tornado diagram"
                    },
                    {
                        day: 4,
                        task: "Compare to other modeling approaches: Implement baseline (e.g., RL, heuristic) | Same scenarios | Measure: prediction accuracy, data efficiency, interpretability"
                    },
                    {
                        day: 5,
                        task: "Create comparison table: Active inference vs alternatives | Dimensions: predictive accuracy, computational cost, data requirements, interpretability | Quantitative + qualitative"
                    },
                    {
                        day: 6,
                        task: "Prepare results package: Organized folder with: code, data, figures, tables, documentation | Version control with git | Push to GitHub (private if needed)"
                    },
                    {
                        day: 7,
                        task: "Write Results section draft: Describe what you found | Include: parameter analysis, model comparisons, key insights | Use figures created this week | Target: 5-7 pages"
                    }
                ]
            },
            {
                title: "Week 32: Advanced Implementation 4",
                dates: "Jun 30 - Jul 6, 2026",
                days: [
                    {
                        day: 1,
                        task: "Conduct validation study: Test model predictions against held-out data | Metrics: prediction error, correlation, classification accuracy | Report quantitative results"
                    },
                    {
                        day: 2,
                        task: "Cross-validation: Implement k-fold CV | Ensures results generalize | Report: mean and std of metrics across folds | Check for overfitting"
                    },
                    {
                        day: 3,
                        task: "Face validity check: Show behavior to domain experts (McDonald, Lee) | Does it look realistic? | Qualitative assessment | Document feedback"
                    },
                    {
                        day: 4,
                        task: "Edge case testing: What breaks the model? | Test: extreme parameters, unusual scenarios | Identify limitations | Document failure modes"
                    },
                    {
                        day: 5,
                        task: "Computational performance analysis: Measure: runtime, memory usage | Profile code | Optimize bottlenecks if needed | Can it run real-time? How large can scenarios be?"
                    },
                    {
                        day: 6,
                        task: "Create supplementary materials: Additional figures, tables, code snippets | Anything that supports main text but too detailed | Could become appendix"
                    },
                    {
                        day: 7,
                        task: "Complete implementation phase: All code done, tested, documented | Results analyzed and written up | Code repository clean and organized | Ready for presentation focus"
                    }
                ]
            },
            {
                title: "Week 33: Presentation Prep 1",
                dates: "Jul 7 - Jul 13, 2026",
                days: [
                    {
                        day: 1,
                        task: "Create presentation outline: Aim for 45 min talk | Sections: Introduction (5min), Background (10min), Methods (10min), Results (15min), Discussion (5min) | List key points"
                    },
                    {
                        day: 2,
                        task: "Build title slide: Title, your name, advisors, date | Introduction slides (3-5): motivation, research questions, contributions | Use compelling figures/images"
                    },
                    {
                        day: 3,
                        task: "Build background slides (8-10): What is active inference? Why driving? Literature overview | Keep it accessible | Anticipate: audience may not know active inference"
                    },
                    {
                        day: 4,
                        task: "Build methods slides (8-10): Model architecture, implementation, experiments | Focus on: key ideas, not every detail | Use diagrams, flowcharts | Make it visual"
                    },
                    {
                        day: 5,
                        task: "Build results slides (10-12): Key findings, figures, tables | One main point per slide | Tell a story | Build anticipation | Use animations strategically"
                    },
                    {
                        day: 6,
                        task: "Build discussion slides (3-5): Implications, limitations, future work | Connect back to research questions | Broader impact | End strong with conclusions"
                    },
                    {
                        day: 7,
                        task: "First complete draft: Put all slides together | Check flow, timing (1-2 min per slide) | Practice once through | Identify: where to slow down, speed up"
                    }
                ]
            },
            {
                title: "Week 34: Presentation Prep 2",
                dates: "Jul 14 - Jul 20, 2026",
                days: [
                    {
                        day: 1,
                        task: "Refine slide design: Consistent fonts, colors, layout | Use university template | High-quality figures | Readable from back of room (test font sizes)"
                    },
                    {
                        day: 2,
                        task: "Create backup slides: Additional details, extra figures, anticipated questions | These go after conclusion | Don't present unless asked"
                    },
                    {
                        day: 3,
                        task: "Practice presentation #1: Time yourself | Record if possible | Watch for: ums, filler words, pacing | Note: slides that need work | Aim for 40-45 min"
                    },
                    {
                        day: 4,
                        task: "Practice presentation #2: Focus on transitions between slides | Make it flow smoothly | Practice explaining complex ideas simply | Anticipate questions"
                    },
                    {
                        day: 5,
                        task: "Practice presentation #3: With a friend/lab mate | Get feedback on: clarity, pacing, interest | What was confusing? What worked well? | Take notes, revise slides"
                    },
                    {
                        day: 6,
                        task: "Prepare for Q&A: List 20 potential questions | Prepare answers | Practice responding | Some will be tough - that's okay! | Admit when you don't know something"
                    },
                    {
                        day: 7,
                        task: "Weekly review: Presentation should be 90% polished | Share slides with McDonald for feedback | Schedule practice talk with lab (next week or week after)"
                    }
                ]
            },
            {
                title: "Week 35: Presentation Prep 3",
                dates: "Jul 21 - Jul 27, 2026",
                days: [
                    {
                        day: 1,
                        task: "Practice presentation #4: Full run-through | Time carefully | Adjust if too long/short | Should be 43-47 min | Leave time for questions"
                    },
                    {
                        day: 2,
                        task: "Practice presentation #5: Focus on difficult parts | Technical details, complex figures | Practice explaining without jargon | Can a non-expert understand?"
                    },
                    {
                        day: 3,
                        task: "Practice presentation #6: Simulate exam conditions | Stand up, present to empty room | Practice as if committee is there | Get comfortable with material"
                    },
                    {
                        day: 4,
                        task: "MOCK PRESENTATION TO LAB: Full formal presentation | Invite: McDonald, Lee, lab mates | Treat like real exam | Get detailed feedback | Take notes"
                    },
                    {
                        day: 5,
                        task: "Incorporate feedback from mock: Revise slides based on comments | What was unclear? What questions did they ask? | Make improvements | May need major revisions"
                    },
                    {
                        day: 6,
                        task: "Practice presentation #7: With all revisions | Check timing again | Should feel smoother | Get comfortable with new slides | Practice Q&A with anticipated questions"
                    },
                    {
                        day: 7,
                        task: "Deep dive on weak areas: Identify 2-3 topics you're least confident on | Study them extra | Prepare detailed explanations | Practice until confident"
                    }
                ]
            },
            {
                title: "Week 36: Presentation Prep 4",
                dates: "Jul 28 - Aug 3, 2026",
                days: [
                    {
                        day: 1,
                        task: "Practice presentation #8: Focus on delivery | Work on: eye contact, voice projection, enthusiasm | Record video | Watch yourself | What can improve?"
                    },
                    {
                        day: 2,
                        task: "Practice presentation #9: Time to add polish | Work on: smooth transitions, explaining figures clearly, engaging storytelling | Make it interesting!"
                    },
                    {
                        day: 3,
                        task: "Practice presentation #10: Full run-through | This should feel natural now | Minor adjustments only | Timing should be perfect | Confident with all material"
                    },
                    {
                        day: 4,
                        task: "Q&A preparation intensive: Practice answering 30 questions | Mix of: clarifying, challenging, extending | Stay calm, think before answering | It's okay to say 'good question, let me think'"
                    },
                    {
                        day: 5,
                        task: "Review literature one final time: Re-read key papers | Refresh memory on details | You might be asked about specific papers | Be prepared to discuss connections"
                    },
                    {
                        day: 6,
                        task: "Practical preparations: Test equipment (laptop, pointer), print backup slides, prepare notes (just in case), confirm room booking, check room tech day before exam"
                    },
                    {
                        day: 7,
                        task: "Final practice #11: One last full run-through | Should be effortless | You're ready! | Light practice only | Don't overdo it | Prepare mentally"
                    }
                ]
            },
            {
                title: "Week 37: Final Polish",
                dates: "Aug 18 - Aug 24, 2026",
                days: [
                    {
                        day: 1,
                        task: "Final slide tweaks: Only minor changes | Fix any typos | Ensure all figures are high-res | Check animations work | Finalize backup slides"
                    },
                    {
                        day: 2,
                        task: "Practice presentation #12: But don't overdo it | Light run-through | Focus on feeling confident | You know this material inside out!"
                    },
                    {
                        day: 3,
                        task: "Practice presentation #13: Focus on introduction and conclusion | These are most important | First impressions and last impressions matter | Make them strong"
                    },
                    {
                        day: 4,
                        task: "Practice presentation #14: Focus on Q&A | Have someone grill you | Practice staying calm under pressure | Think before answering | It's okay to pause"
                    },
                    {
                        day: 5,
                        task: "Practice presentation #15: Final full run-through | This is it! | Record time: should be 43-47 min | Smooth, confident, polished | You're ready!"
                    },
                    {
                        day: 6,
                        task: "Mental preparation: Visualize success | Review: why you're qualified, what you've accomplished | Prepare: what to wear, what to bring | Get good sleep | Eat well"
                    },
                    {
                        day: 7,
                        task: "Rest day: Light review only | Read through notes | Don't practice full presentation | Relax | Trust your preparation | You've got this! 🎉"
                    }
                ]
            },
            {
                title: "Week 38: EXAM WEEK! 🎓",
                dates: "Aug 25 - Sept 1, 2026",
                days: [
                    {
                        day: 1,
                        task: "Equipment check: Test laptop, backup laptop, HDMI adapter, pointer | Print backup slides | Review slides on different device | Ensure everything works"
                    },
                    {
                        day: 2,
                        task: "Final Q&A prep: Review list of potential questions | Practice answers one more time | Stay calm | Remember: committee wants you to succeed"
                    },
                    {
                        day: 3,
                        task: "Light review: Read through notes | Don't practice full presentation | Just refresh memory | Stay relaxed | Conserve energy"
                    },
                    {
                        day: 4,
                        task: "Day before exam: Visit room, test tech | Do ONE light run-through | Prepare: clothes, materials, water bottle | Early bedtime | Sleep is crucial!"
                    },
                    {
                        day: 5,
                        task: "EXAM DAY! 🎓 Wake up early | Eat good breakfast | Arrive 30min early | Test tech one final time | Deep breaths | You're prepared | GO ACE IT! 🌟"
                    },
                    {
                        day: 6,
                        task: "Post-exam celebration: You did it! 🎉 | Reflect on what went well | Note areas for improvement (for future defense) | Thank committee | Update LinkedIn!"
                    },
                    {
                        day: 7,
                        task: "🏆 CONGRATULATIONS DR.-TO-BE! You're qualified! | Update tracker to COMPLETE | Reflect on 10-month journey | What did you learn? | Plan dissertation next steps!"
                    }
                ]
            }
        ]
    }
};
