<h1 align="center">
LLMOps Course on Databricks
</h1>

## Project Proposal
The Root Cause Analysis for e-comerce Our project aims to deliver an automated Root Cause Analysis (RCA) engine built on Databricks Genie to detect and explain anomalies in order mix and volume within the Olist e-commerce dataset. By integrating a reasoning agent framework, the system decomposes performance shifts into structured drivers such as lost customers, category mix shifts, or concentration spikes. 

## Practical information
- Weekly lectures on Wednesdays 16:00-18:00 CET.
- Weekly Q&A on Mondays 16:00-17:00 CET.
- Code for the lecture is shared before the lecture.
- Presentation and lecture materials are shared right after the lecture.
- Video of the lecture is uploaded within 24 hours after the lecture.

- Every week we set up a deliverable, and you implement it with your own dataset.
- To submit the deliverable, create a feature branch in that repository, and a PR to main branch. The code can be merged after we review & approve & CI pipeline runs successfully.
- The deliverables can be submitted with a delay (for example, lecture 1 & 2 together), but we expect you to finish all assignments for the course before the demo day.


## Set up your environment
In this course, we use serverless environment 4, which uses Python 3.12.
In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

To create a new environment and create a lockfile, run:

```
uv sync --extra dev
```



