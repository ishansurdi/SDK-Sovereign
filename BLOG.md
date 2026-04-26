# SDK-Sovereign: How This Idea Started, Broke, and Slowly Became Real

If I told this story in the polished way, it would sound very simple.

I had an idea for a multi-agent environment. I built it. I trained two role-specific agents. I added a demo, some plots, and a notebook. Then it all worked.

That is not what happened.

SDK-Sovereign came from a feeling I could not shake: most AI demos did not feel real to me. They looked neat, but they did not feel like engineering. They felt like controlled theater. A benchmark, an output, a graph, and then nothing underneath that really carried the weight of a real operational problem.

I wanted to build something that felt closer to panic.

The idea that kept pulling me back was this: what if an engineering team woke up one day and a core SDK they depended on was suddenly no longer acceptable? Not as a policy debate. As a direct product problem. Payments break. Maps break. Messaging breaks. Users are still there. Deadlines are still there. You still have to ship. And the people involved do not all get to see the whole system.

That was the seed of SDK-Sovereign.

If you want the short version of what is actually in the project, it is this:

- a live Hugging Face Space demo
- an OpenEnv environment with two asymmetric roles
- a hardened Colab notebook using Unsloth and TRL
- real reward and loss plots from an actual run
- a README that explains the environment, the reward logic, and the results honestly

From the beginning, I did not want one magical agent with perfect information. That felt dishonest. Real systems are full of boundaries. Real teams are full of partial visibility. So I split the task into two roles.

The Integration Lead can see the code and the error log, because that person is trying to make the system work again.

The Security Auditor can see the sovereign allow-list and the negotiation history, but not the code, because that person is protecting constraints the implementer might be tempted to ignore.

That split ended up being the heart of the project. Not the model choice, not the UI, not the notebook. That asymmetry. That feeling that neither side is powerful enough on its own, and that coordination under incomplete information is the actual task.

At the ideation stage, this all felt sharp in my head. In the repo, it was much blurrier.

I had the concept before I had the shape. I knew what I wanted it to feel like, but not yet how to make every piece reinforce that feeling. That became the pattern for most of this project. I was not just coding features. I was constantly trying to align the environment, the verifier, the training loop, the demo, and the write-up so they all told the same truth.

The first truth I kept colliding with was honesty.

It is very easy to make something like this look more capable than it really is. I could have hidden the difference between rule-based behavior and model-based behavior. I could have made the UI sound more confident than the backend deserved. I could have shown reward movement and let people quietly assume that meant the benchmark was solved.

I did not want that.

So I kept pushing the project toward honesty, even when it made the output less flattering. If trained mode was not really loaded, I wanted that to be visible. If rule mode was still the most reliable mode, I wanted that to be visible. If the model was doing something interesting but not actually solving the task, I wanted the plots to make that obvious too.

That decision made the project harder, but also made it feel worth building.

Then came the training reality check.

The fantasy version in my head was elegant: define the environment, create the roles, let RL discover useful behavior, and watch coordination emerge. The real version was much more stubborn. Baseline behavior was weak. Rewards sometimes improved in ways that looked promising, but actual task completion did not necessarily move with them. The notebook became the place where everything converged, and that meant every Colab failure felt like it was attacking the whole project at once.

There were moments where I felt like I was not really training a system so much as negotiating with an unreliable machine stack. Runtime issues. Persistence issues. Warnings everywhere. Upload failures. Sessions disconnecting. Small configuration mistakes eating huge chunks of time. Even when the GPU was finally attached and the model loaded, I still had to ask whether the thing was genuinely learning the task or just learning how to look active.

That was probably the hardest emotional part of the project.

Because there is a particular kind of frustration when the build is alive enough to consume hours, but not reliable enough to reward them cleanly.

Eventually I changed my approach.

I stopped trying to make the notebook elegant. I started trying to make it survivable.

That changed everything.

I pushed the pipeline toward something more practical and more honest about the time constraints: low steps, fast supervised bootstrap, tiny RL refinement, constant logging, aggressive checkpoint persistence, frequent uploads to Hugging Face, W&B tracking, JSONL traces, final bundles, and plots that could answer judge questions quickly.

At some point the goal quietly changed from “train a great model” to “if this runtime dies in twenty minutes, what proof do I still have?”

That shift sounds small, but it completely changed how I thought about the project. In a hackathon setting, persistence is not a side feature. It is part of the work. If the results disappear when the GPU goes away, then the project is much less real than it looks.

And the GPU did fight back.

One of the more annoying moments was hitting the Unsloth error about not finding a torch accelerator. It was not some deep research failure. It was just Colab being on CPU when the whole stack needed GPU. But those moments matter because they drain momentum. So I patched the notebook to fail early and clearly, instead of forcing whoever runs it to decipher a low-level stack trace.

That became another theme of the build: reducing stupid pain.

Not everything hard was intellectually interesting. Some of it was just friction. Some of it was the demo not lining up with the backend. Some of it was the verifier needing to be hardened so the Space would behave consistently. Some of it was test and route coverage that I knew I needed if I wanted to trust the project later. Some of it was documentation that technically explained the project but did not sound like a human being had lived through making it.

And then there was the most uncomfortable part: the results themselves.

I got the system to train. I got the adapters saved. I got the logs, plots, and artifacts produced. I got the live demo and the notebook into a much more robust state. But even with that progress, the eval told a difficult truth: reward could improve while pass rate stayed flat.

That matters a lot to me.

Because I did not want to write a fake success story. I wanted to build something I could look at directly, even if the answer was inconvenient.

The inconvenient answer is that the project clearly moved forward, but it did not magically become solved.

## What Actually Improved

- Mean reward moved from `-0.583` to `+0.194`
- Pass rate stayed at `0.0%`
- The trained agents became more structured, but not yet reliably successful end to end
- The pipeline became much more robust, inspectable, and reproducible

That combination is awkward, but I trust it more than a fake clean win.

If everything had worked immediately, this would have become another polished AI artifact that says all the right words and leaves no room for doubt. But doubt is part of this build. Resistance is part of this build. I kept having to ask myself what exactly I was proving, whether the metrics actually meant what I wanted them to mean, and whether I was improving the system or just improving its presentation.

That question kept me from taking shortcuts I would not respect later.

Why did I keep the split-role structure instead of collapsing everything into one stronger agent? Because the split is the idea.

Why did I keep baseline versus trained comparisons explicit? Because otherwise I would be tempted to narrate progress I could not defend.

Why did I spend time on smoke tooling, API checks, route tests, persistence, and demo behavior even while the training story was still incomplete? Because weights are not the whole project. Trust is the whole project.

And that is really what this turned into for me.

SDK-Sovereign stopped being just an environment idea. It became a test of whether I could build something ambitious under time pressure without lying to myself about what was working.

I still love the core instinct behind it. I still think this is where more interesting AI evaluation should go: toward settings with partial information, institutional constraints, operational urgency, and outcomes that can actually be checked. Not just clever outputs. Not just vibes. Something closer to real systems pressure.

What I have right now is not perfect. It still needs better curriculum, stronger completion behavior, and training signal that maps more directly to success. But it feels real now. It feels like something that has been wrestled into shape instead of merely assembled.

That is the part I am most proud of.

And if I kept building from here, that is what I would push next: stronger completion behavior, better curriculum design, and reward shaping that closes the gap between “more structured” and “actually solved.”

## Where To Look

- Space: [ishansurdi/SDK-Sovereign](https://huggingface.co/spaces/ishansurdi/SDK-Sovereign)
- Live demo: [ishansurdi-sdk-sovereign.hf.space/play](https://ishansurdi-sdk-sovereign.hf.space/play)
- README: ./README.md
- Training notebook: ./notebooks/00_hardened_pipeline.ipynb
