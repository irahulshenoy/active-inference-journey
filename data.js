const learningPlan = {
    tonight: {
        title: "Tonight's Task - Your First Checkpoint! ⭐",
        description: "Get started right now!",
        tasks: [
            "30 min: Watch 'PyTorch in 100 seconds' + 'PyTorch for Absolute Beginners'",
            "30 min: Install Anaconda + PyTorch, run your first tensor operation",
            "10 min: Create a learning journal (can be simple text file or notion)",
            "Tomorrow: Start Week 1, Day 1 properly!"
        ]
    },
    
    phase1: {
        title: "PHASE 1: FOUNDATIONS",
        dates: "Now - Jan 20, 2026",
        duration: "12 weeks | 14 hours/week | 2 hours/day",
        weeks: [
            {
                title: "Week 1: Python Basics",
                dates: "Nov 25 - Dec 1, 2025",
                days: [
                    { day: 1, task: "Install Python/Anaconda, VS Code setup, first 'Hello World'" },
                    { day: 2, task: "Variables, data types, basic operations" },
                    { day: 3, task: "Lists, tuples, dictionaries" },
                    { day: 4, task: "If/else, loops (for, while)" },
                    { day: 5, task: "Functions, basic modules" },
                    { day: 6, task: "Practice problems from Python Crash Course" },
                    { day: 7, task: "Weekly review + build a simple calculator" }
                ]
            },
            {
                title: "Week 2: NumPy + Math Refresh",
                dates: "Dec 2 - Dec 8, 2025",
                days: [
                    { day: 1, task: "NumPy arrays, indexing, slicing" },
                    { day: 2, task: "Array operations, broadcasting" },
                    { day: 3, task: "Matrix operations (review linear algebra concepts)" },
                    { day: 4, task: "Vector operations, dot products" },
                    { day: 5, task: "Probability basics review (Khan Academy)" },
                    { day: 6, task: "Practice: Implement basic statistics (mean, std, covariance)" },
                    { day: 7, task: "Weekly project: Matrix manipulations" }
                ]
            },
            {
                title: "Week 3: Data Handling",
                dates: "Dec 9 - Dec 15, 2025",
                days: [
                    { day: 1, task: "Pandas basics (DataFrames, Series)" },
                    { day: 2, task: "Reading CSV files, data manipulation" },
                    { day: 3, task: "Matplotlib basics (you know ggplot, this will be easy!)" },
                    { day: 4, task: "Seaborn for statistical plots" },
                    { day: 5, task: "Practice: Load driving data, make plots" },
                    { day: 6, task: "Probability distributions with scipy" },
                    { day: 7, task: "Weekly project: Analyze and visualize a dataset" }
                ]
            },
            {
                title: "Week 4: Math Deep Dive",
                dates: "Dec 16 - Dec 22, 2025",
                days: [
                    { day: 1, task: "Calculus refresher: derivatives, chain rule" },
                    { day: 2, task: "Gradients, partial derivatives" },
                    { day: 3, task: "Probability review: Bayes' theorem" },
                    { day: 4, task: "Conditional probability, distributions" },
                    { day: 5, task: "Linear algebra: eigenvalues, eigenvectors" },
                    { day: 6, task: "3Blue1Brown videos on linear algebra + calculus" },
                    { day: 7, task: "Weekly review: Work through math problems" }
                ]
            },
            {
                title: "Week 5: PyTorch Basics",
                dates: "Dec 23 - Dec 29, 2025",
                days: [
                    { day: 1, task: "Install PyTorch, understand tensors" },
                    { day: 2, task: "Tensor operations, moving to/from NumPy" },
                    { day: 3, task: "Autograd: automatic differentiation" },
                    { day: 4, task: "Building a simple neural network (nn.Module)" },
                    { day: 5, task: "Loss functions, optimizers" },
                    { day: 6, task: "Training loop basics" },
                    { day: 7, task: "Project: Linear regression in PyTorch" }
                ]
            },
            {
                title: "Week 6: Neural Networks",
                dates: "Dec 30, 2025 - Jan 5, 2026",
                days: [
                    { day: 1, task: "Multi-layer perceptrons" },
                    { day: 2, task: "Activation functions (ReLU, sigmoid, softmax)" },
                    { day: 3, task: "Forward and backward propagation" },
                    { day: 4, task: "Overfitting, regularization" },
                    { day: 5, task: "Train/validation/test splits" },
                    { day: 6, task: "Project: Classify MNIST digits" },
                    { day: 7, task: "Weekly review + debug your classifier" }
                ]
            },
            {
                title: "Week 7: Probability in PyTorch",
                dates: "Jan 6 - Jan 12, 2026",
                days: [
                    { day: 1, task: "torch.distributions module" },
                    { day: 2, task: "Categorical distributions" },
                    { day: 3, task: "Normal (Gaussian) distributions" },
                    { day: 4, task: "Sampling from distributions" },
                    { day: 5, task: "Log probabilities, KL divergence" },
                    { day: 6, task: "Practice: Implement probabilistic models" },
                    { day: 7, task: "Project: Probabilistic classifier" }
                ]
            },
            {
                title: "Week 8: Reinforcement Learning Basics",
                dates: "Jan 13 - Jan 19, 2026",
                days: [
                    { day: 1, task: "MDP basics: states, actions, rewards" },
                    { day: 2, task: "Value functions, Q-learning concepts" },
                    { day: 3, task: "Policy gradients introduction" },
                    { day: 4, task: "Watch David Silver RL lectures (Lecture 1-2)" },
                    { day: 5, task: "Implement simple grid world" },
                    { day: 6, task: "Q-learning in PyTorch" },
                    { day: 7, task: "Checkpoint: Review all Phase 1 material" }
                ]
            }
        ]
    },
    
    phase2: {
        title: "PHASE 2: ACTIVE INFERENCE DEEP DIVE",
        dates: "Jan 20 - May 25, 2026",
        duration: "16 weeks | 20 hours/week | ~3 hours/day",
        weeks: [
            {
                title: "Week 9: Introduction to Active Inference",
                dates: "Jan 20 - Jan 26, 2026",
                days: [
                    { day: 1, task: "Read Parr book Chapter 1 (Introduction)" },
                    { day: 2, task: "Read Parr Chapter 2 (Neuroscience background)" },
                    { day: 3, task: "Watch Karl Friston introductory lecture on YouTube" },
                    { day: 4, task: "Read Parr Chapter 3 (Bayesian brain)" },
                    { day: 5, task: "Notes: Create concept map of key ideas" },
                    { day: 6, task: "Watch Active Inference Institute intro videos" },
                    { day: 7, task: "Start qualifying exam literature tracker spreadsheet" }
                ]
            },
            {
                title: "Week 10: Free Energy Principle",
                dates: "Jan 27 - Feb 2, 2026",
                days: [
                    { day: 1, task: "Read Parr Chapter 4 (Free energy)" },
                    { day: 2, task: "Work through free energy equation derivations" },
                    { day: 3, task: "Read Bogacz (2017) tutorial paper" },
                    { day: 4, task: "Implement free energy calculation in Python" },
                    { day: 5, task: "Read Parr Chapter 5 (Message passing)" },
                    { day: 6, task: "Practice: Calculate free energy for simple examples" },
                    { day: 7, task: "Weekly review: Can you explain free energy to someone?" }
                ]
            },
            {
                title: "Week 11: Generative Models",
                dates: "Feb 3 - Feb 9, 2026",
                days: [
                    { day: 1, task: "Read Parr Chapter 6 (POMDPs)" },
                    { day: 2, task: "Implement simple POMDP in Python" },
                    { day: 3, task: "A, B, C, D matrices explained" },
                    { day: 4, task: "Read pymdp documentation thoroughly" },
                    { day: 5, task: "Install pymdp, run basic examples" },
                    { day: 6, task: "Implement gridworld with pymdp" },
                    { day: 7, task: "Project: Build your own simple generative model" }
                ]
            },
            {
                title: "Week 12: Active Inference for Action",
                dates: "Feb 10 - Feb 16, 2026",
                days: [
                    { day: 1, task: "Read Parr Chapter 7 (Planning and policy selection)" },
                    { day: 2, task: "Expected free energy (EFE) explained" },
                    { day: 3, task: "Implement EFE calculation" },
                    { day: 4, task: "Read pymdp tutorial 'Active Inference from Scratch'" },
                    { day: 5, task: "Code along with the tutorial" },
                    { day: 6, task: "Modify tutorial for different scenario" },
                    { day: 7, task: "Checkpoint: Can you build an active inference agent?" }
                ]
            },
            {
                title: "Week 13: McDonald's Papers - Deep Dive",
                dates: "Feb 17 - Feb 23, 2026",
                days: [
                    { day: 1, task: "Read Wei et al. (2024) - AV takeovers paper" },
                    { day: 2, task: "Read Wei et al. (2022) - Modeling driver responses" },
                    { day: 3, task: "Read Engström et al. (2024) - Adaptive driving" },
                    { day: 4, task: "Create detailed notes on each paper's methods" },
                    { day: 5, task: "Identify common modeling patterns" },
                    { day: 6, task: "Add papers to qualifying exam literature review" },
                    { day: 7, task: "Write summary: How is active inference applied to driving?" }
                ]
            },
            {
                title: "Week 14: Driver Behavior Modeling",
                dates: "Feb 24 - Mar 2, 2026",
                days: [
                    { day: 1, task: "Read Parr Chapter 9 (Continuous time active inference)" },
                    { day: 2, task: "Review car-following models (IDM, OVM)" },
                    { day: 3, task: "How does active inference extend these models?" },
                    { day: 4, task: "Read additional driver behavior papers from proposal" },
                    { day: 5, task: "Implement simple car-following in Python" },
                    { day: 6, task: "Add active inference to car-following model" },
                    { day: 7, task: "Project: Simulate driver following behavior" }
                ]
            },
            {
                title: "Week 15: Lab Code Integration",
                dates: "Mar 3 - Mar 9, 2026",
                days: [
                    { day: 1, task: "Meet with McDonald: Get existing code" },
                    { day: 2, task: "Understand code structure, dependencies" },
                    { day: 3, task: "Run existing examples" },
                    { day: 4, task: "Debug any issues with lab mates" },
                    { day: 5, task: "Modify code for new scenario" },
                    { day: 6, task: "Experiment with different parameters" },
                    { day: 7, task: "Document what you learned" }
                ]
            },
            {
                title: "Week 16: Trust and Takeover Modeling",
                dates: "Mar 10 - Mar 16, 2026",
                days: [
                    { day: 1, task: "Read evidence accumulation models (drift diffusion)" },
                    { day: 2, task: "How does active inference relate to EA models?" },
                    { day: 3, task: "Read Lee & See (2004) - Trust in automation" },
                    { day: 4, task: "Implement trust dynamics in active inference" },
                    { day: 5, task: "Model driver takeover decision" },
                    { day: 6, task: "Add all new papers to lit review" },
                    { day: 7, task: "Mid-point check: Do mock presentation to lab mate" }
                ]
            },
            {
                title: "Week 17: Preference Modeling",
                dates: "Mar 17 - Mar 23, 2026",
                days: [
                    { day: 1, task: "Re-read proposal Thrust 1 in detail" },
                    { day: 2, task: "How to operationalize preferences in active inference?" },
                    { day: 3, task: "Prior preferences (C matrices) explained" },
                    { day: 4, task: "Implement preference learning" },
                    { day: 5, task: "Read IRL papers from proposal" },
                    { day: 6, task: "Compare active inference to IRL for preferences" },
                    { day: 7, task: "Project: Model individual driver preferences" }
                ]
            },
            {
                title: "Week 18: Hierarchical Models",
                dates: "Mar 24 - Mar 30, 2026",
                days: [
                    { day: 1, task: "Read Parr Chapter 11 (Structure learning)" },
                    { day: 2, task: "Hierarchical active inference explained" },
                    { day: 3, task: "Multi-scale modeling (micro to macro)" },
                    { day: 4, task: "How does individual behavior scale to traffic?" },
                    { day: 5, task: "Read traffic flow theory papers from proposal" },
                    { day: 6, task: "Implement multi-agent simulation" },
                    { day: 7, task: "Weekly review: Connection to qualifying exam topic" }
                ]
            },
            {
                title: "Week 19: Deep Active Inference",
                dates: "Mar 31 - Apr 6, 2026",
                days: [
                    { day: 1, task: "Read papers on deep active inference" },
                    { day: 2, task: "Combining neural networks with active inference" },
                    { day: 3, task: "Variational autoencoders (VAE) review" },
                    { day: 4, task: "Implement VAE in PyTorch" },
                    { day: 5, task: "Connect VAE concepts to active inference" },
                    { day: 6, task: "Read about world models" },
                    { day: 7, task: "Project: Neural network generative model" }
                ]
            },
            {
                title: "Week 20: Literature Synthesis",
                dates: "Apr 7 - Apr 13, 2026",
                days: [
                    { day: 1, task: "Review all papers collected so far (~30-40)" },
                    { day: 2, task: "Organize by theme (preference, trust, traffic, etc.)" },
                    { day: 3, task: "Identify gaps in literature" },
                    { day: 4, task: "Create concept map connecting papers" },
                    { day: 5, task: "Write draft outline of qualifying exam" },
                    { day: 6, task: "Get feedback from McDonald on outline" },
                    { day: 7, task: "Checkpoint: 50% done with lit review" }
                ]
            },
            {
                title: "Week 21-22: Reproduce Lab Results",
                dates: "Apr 14 - Apr 27, 2026",
                days: [
                    { day: "1-7", task: "Work through paper implementation" },
                    { day: "8-14", task: "Validate results, understand every detail" }
                ]
            },
            {
                title: "Week 23: Nudging & Boosting",
                dates: "Apr 28 - May 4, 2026",
                days: [
                    { day: "1-3", task: "Read behavioral science papers on nudging" },
                    { day: "4-5", task: "How to model in active inference framework?" },
                    { day: "6-7", task: "Implement preference shaping mechanisms" }
                ]
            },
            {
                title: "Week 24: Traffic Simulation",
                dates: "May 5 - May 11, 2026",
                days: [
                    { day: "1-3", task: "Learn traffic simulation tools (SUMO/ProjectChrono)" },
                    { day: "4-7", task: "Integrate active inference agents into traffic sim" }
                ]
            }
        ]
    },
    
    phase3: {
        title: "PHASE 3: MASTERY & EXAM PREP",
        dates: "May 26 - Sept 1, 2026",
        duration: "15 weeks | 10-14 hours/week",
        weeks: [
            {
                title: "Week 25: Complete Literature Review (Part 1)",
                dates: "May 26 - Jun 1, 2026",
                days: [
                    { day: "1-7", task: "Finish reading all cited papers in proposal (50-60 papers)" }
                ]
            },
            {
                title: "Week 26: Lit Review - Section 1",
                dates: "Jun 2 - Jun 8, 2026",
                days: [
                    { day: "1-7", task: "Write lit review section 1: Active inference foundations" }
                ]
            },
            {
                title: "Week 27: Lit Review - Section 2",
                dates: "Jun 9 - Jun 15, 2026",
                days: [
                    { day: "1-7", task: "Write lit review section 2: Driver behavior & preferences" }
                ]
            },
            {
                title: "Week 28: Lit Review - Section 3",
                dates: "Jun 16 - Jun 22, 2026",
                days: [
                    { day: "1-7", task: "Write lit review section 3: Multi-scale traffic interactions" }
                ]
            },
            {
                title: "Week 29-32: Advanced Implementation",
                dates: "Jun 23 - Jul 20, 2026",
                days: [
                    { day: "Week 29", task: "Build complete driver-AV interaction model" },
                    { day: "Week 30", task: "Implement preference learning from data" },
                    { day: "Week 31", task: "Create visualizations of model behavior" },
                    { day: "Week 32", task: "Run parameter sensitivity analyses + Compare to other modeling approaches" }
                ]
            },
            {
                title: "Week 33: Presentation Outline",
                dates: "Jul 21 - Jul 27, 2026",
                days: [
                    { day: "1-7", task: "Create presentation outline" }
                ]
            },
            {
                title: "Week 34: Build Slides",
                dates: "Jul 28 - Aug 3, 2026",
                days: [
                    { day: "1-7", task: "Build slides (intro, methods, results, discussion)" }
                ]
            },
            {
                title: "Week 35: Practice Presentation",
                dates: "Aug 4 - Aug 10, 2026",
                days: [
                    { day: "1-7", task: "Practice presentation 3x" }
                ]
            },
            {
                title: "Week 36: Mock Presentation",
                dates: "Aug 11 - Aug 17, 2026",
                days: [
                    { day: "1-7", task: "Mock presentation to lab, get feedback" }
                ]
            },
            {
                title: "Week 37: Final Revisions",
                dates: "Aug 18 - Aug 24, 2026",
                days: [
                    { day: "1-7", task: "Revise based on feedback, practice 5 more times" }
                ]
            },
            {
                title: "Week 38: Final Prep & QUALIFY!",
                dates: "Aug 25 - Sept 1, 2026",
                days: [
                    { day: "1-6", task: "Final prep, relaxation" },
                    { day: 7, task: "🎉 ACE YOUR QUALIFYING EXAM! 🎉" }
                ]
            }
        ]
    }
};
