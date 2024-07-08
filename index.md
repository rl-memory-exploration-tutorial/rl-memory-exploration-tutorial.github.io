---
layout: home  
title: Unlocking Exploration 
subtitle: Self-Motivated Agents Thrive on Memory-Driven Curiosity  
---


This webpage serves as the companion site for our [**AAMAS24 tutorial**](https://www.aamas2024-conference.auckland.ac.nz/accepted/tutorials/){:target="_blank"}, developed by Hung Le, Hoang Nguyen, and Dai Do. It features notes designed to complement our tutorial, all crafted using Colab notebooks for your interactive exploration. Additionally, you'll find links to our video presentation and slides. Please note that this webpage will continue to evolve until the official tutorial date (May 5, 2024), and may undergo further updates thereafter.

#### Duration

Half day. Time and place: 
- 8:30-12:00, 06 May, 2024
-  Cordis Hotel, AUCKLAND, NEW ZEALAND 

#### Abstract

Despite remarkable successes in various domains such as robotics and games, Reinforcement Learning (RL) still struggles with exploration inefficiency.  For example, in hard Atari games, state-of-the-art agents often require billions of trial actions, equivalent to years of practice, while a moderately skilled human player can achieve the same score in just a few hours of play. This contrast emerges from the difference in exploration strategies between humans, leveraging memory, intuition and experience, and current RL agents, primarily relying on random trials and errors. This tutorial reviews recent advances in enhancing RL exploration efficiency through intrinsic motivation or curiosity, allowing agents to navigate environments without external rewards. Unlike previous surveys, we analyze intrinsic motivation through a memory-centric perspective, drawing parallels between human and agent curiosity, and providing a memory-driven taxonomy of intrinsic motivation approaches.

The talk consists of three main parts. Part A provides a brief introduction to RL basics, delves into the historical context of the explore-exploit dilemma, and raises the challenge of exploration inefficiency. In Part B, we present a taxonomy of self-motivated agents leveraging deliberate, RAM-like, and replay memory models to compute surprise, novelty, and goal, respectively. Part C explores advanced topics, presenting recent methods using language models and causality for exploration. Whenever possible, case studies and hands-on coding demonstrations.
will be presented.

#### Target Audience

This tutorial is mainly designed for students and academics who work on Reinforcement Learning. It is also open to research engineers and industry practitioners who need to apply efficient reinforcement learning in their jobs. Basic familiarity with reinforcement learning is assumed, and an additional understanding of deep learning and neural networks would be beneficial. No special equipment is required, but attendees are encouraged to bring their laptops to experiment with models hands-on.

#### Expected Gain

1. **Technical Understanding:** Participants will gain a deeper understanding of exploration inefficiency in RL and how intrinsic motivation can address this challenge.

2. **Practical Skills:** Attendees will acquire practical skills in implementing several memory-driven curiosity methods for efficient exploration, through hands-on coding demonstrations.

#### Tutorial Outline

- [Part A: Reinforcement LearningFundamentals and Exploration Inefficiency](./parta.md)
  + Welcome and Introduction (5 minutes)
     * Overview of the tutorial
     * Brief speaker introductions
  + Reinforcement Learning Basics (10 minutes)
     * Key components and frameworks
     * Classic exploration 
  + Exploring Challenges in Deep RL (10 minutes)
     * Hard exploration problems
     * Simple exploring solutions 
  + QA and Demo (5 minutes)
  
- [Part B: Intrinsic Motivation, Surprise and Novelty]
  + Principles and Frameworks (10 minutes)
     * Reward shaping and the role of memory
     * A taxonomy of memory-driven intrinsic exploration
  + [Deliberate Memory for Surprise-driven Exploration](./partb_surprise_classic.md) (25 minutes)
     * [Forward dynamics prediction](./partb_surprise_atari.md) 
	 * Advanced dynamics-based surprises
     * Ensemble and disagreement
  + Break (20 minutes)
  + [RAM-like Memory for Novelty-based Exploration](./partb_novelty.md) (25 minutes)
     * Count-based memory 
     * Episodic memory 
	 * Hybrid memory 
  + Replay Memory (20 minutes)
     * Novelty-based replay
     * Performance-based replay 
  + QA and Demo (10 minutes)
  
- [Part C: Advanced Topics](./partc.md)
  + Language-guided exploration (20 minutes)
     * Language-assisted RL 
	 * LLM-based exploration
  + Causal discovery for exploration (20 minutes)
     * Statistical approaches 
	 * Deep learning approaches
  + Closing Remarks (10 minutes)
  + QA and Demo (10 minutes)

#### Tutorial Materials

- [Tutorial slides](https://www.slideshare.net/slideshow/unlocking-exploration-self-motivated-agents-thrive-on-memory-driven-curiosity/268070526)  
- [Tutorial proposal](./main.pdf)
- [Tutorial code](https://github.com/rl-memory-exploration-tutorial/rl-memory-exploration-tutorial.github.io/tree/main/resources/code)
- Blogs:
  + [Part A] (https://hungleai.substack.com/p/curious-agents-saga-part-1)
  + [Part B] (https://hungleai.substack.com/p/curious-agents-saga-part-2)
  + [Part B, C] (https://hungleai.substack.com/p/curious-agents-saga-part-3) 
