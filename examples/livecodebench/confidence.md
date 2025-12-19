## Step 1: Generate the Initial Population (Iteration 1)

This step focuses on creating a high-quality initial "gene pool" of solutions using the **Confidence-Gated Escalation** strategy.

* **How it works:**
    1.  A small, cost-effective LLM is used by default to generate the initial set of candidate programs.
    2.  For each candidate solution, the system monitors its `Lowest Group Confidence` in real-time.
    3.  If this confidence score drops below a pre-set threshold at any point, it signals that the problem's complexity is high.
    4.  Generation for that specific candidate is immediately halted, and the task (including the prompt and partially generated code) is escalated to a large, high-capability LLM to complete.

* **Impact:** This ensures your starting population is high-quality. You save costs on simpler problems while preventing flawed initial solutions on complex ones, giving the subsequent 31 iterations a much better foundation.

***

## Step 2: Mutate and Refine the Population (Iterations 2-32)

In the core evolutionary loop, this step uses **Context-Aware Routing for Granular Mutations** to intelligently refine existing solutions.

* **How it works:**
    1.  When a parent program is selected for mutation, the system first analyzes the `Tail Confidence` of the code right before the mutation point.
    2.  If the score is low (indicating complex code), the large LLM is chosen directly for the surgical mutation task.
    3.  If the score is high (indicating simple code), the small LLM attempts the mutation. During its attempt, its `Lowest Group Confidence` is monitored. If it drops, the task is escalated to the large LLM.

* **Impact:** This protects high-quality, evolved programs from being corrupted by a less capable model. It intelligently allocates the powerful model to the most challenging refinement tasks, increasing the probability of beneficial mutations, especially in the later, more complex stages of the run.

***

## Step 3: Adapt the Overall Strategy (Across Iterations)

This final meta-strategy, **Population-Level Confidence for Dynamic Default Model Selection**, allows the system to adapt as the solutions evolve and the problem's difficulty changes.

* **How it works:**
    1.  The system logs confidence scores and which model was used for all successful programs.
    2.  Periodically (e.g., every 8 iterations), it reviews this population-level data.
    3.  If it detects a trend—such as new, high-performing programs increasingly requiring escalation to the large LLM—it signals that the evolutionary search has entered a more complex phase.
    4.  In response, the system can automatically shift the **default model** from the small LLM to the large LLM for the next block of iterations.

* **Impact:** This makes your entire strategy adaptive. The system might rely heavily on the small LLM for the first ~10 iterations to find a basic solution, then proactively switch to defaulting to the large LLM for iterations 11-32 as the task shifts to complex optimization, ensuring you always use the right tool for the job.