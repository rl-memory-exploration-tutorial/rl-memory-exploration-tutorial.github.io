---
layout: home  
title: Unlocking Exploration 
subtitle: Self-Motivated Agents Thrive on Memory-Driven Curiosity  
---

This webpage serves as the companion site for our [**AAMAS24 tutorial**](https://www.aamas2024-conference.auckland.ac.nz/accepted/tutorials/){:target="_blank"}, developed by Hung Le, Hoang Nguyen, and Dai Do. It features notes designed to complement our tutorial, all crafted using Colab notebooks for your interactive exploration. Additionally, you'll find links to our video presentation and slides. Please note that this webpage will continue to evolve until the official tutorial date (May 5, 2024), and may undergo further updates thereafter.

#### Duration

Half day. Time and place: TBU. 

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

- [Part A](./parta.md)
- [Part B](./partb-surprise.md)
  + [Surprise](./partb-surprise.md)
  + [Novelty](./partb-novelty.md)
  + [Replay](./partb-replay.md)
- [Part C](./partc-language.md)
  + [Language](./partc-language.md)
  + [Causality](./partc-causality.md)

#### Tutorial Materials

- [Tutorial video](./) 
- [Tutorial slides](./)  
- [Tutorial proposal](./main.pdf)
- [Tutorial code](./)
