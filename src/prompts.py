# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Prompts
#
# Task-specific instruction strings for all baselines × benchmarks.

# %%
from juplit import test

# %%
PHANTOM_WIKI_REACT_INSTRUCTIONS = """
The final answer must follow these rules:
  - Multiple answers: comma-separated, e.g. "Alice Smith,Bob Jones"
  - No answer found: empty string ""
  - Numerical answer: just the number, e.g. "4" or "6,5" for multiple
  - Numerical answer not found: "0"

Here are some examples:

---
Task: "Who is the sister of Aida Wang?" 

I need to retrieve the article about Aida Wang and find who their sister is.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Aida Wang"}
}
Observation: "# Aida Wang ## Family The sisters of Aida Wang are Barabara Beltran, Vicki Hackworth.  The mother of Aida Wang is Shelli
Beltran.  The father of Aida Wang is Dino Beltran.  The daughter of Aida Wang is Johnetta Wang.  The husband of Aida Wang is Ryan Wang.
## Friends The friends of Aida Wang are Alvaro Smock, Lannie Smock, Leslee Toombs, Ryan Wang.  ## Attributes The date of birth of Aida
Wang is 0985-05-30.  The occupation of Aida Wang is personal assistant.  The hobby of Aida Wang is meditation.  The gender of Aida Wang
is female."

The sisters of Aida Wang are Barabara Beltran, Vicki Hackworth. So the answer is Barabara Beltran, Vicki Hackworth.
Action:
{
"name": "final_answer",
"arguments": {"answer": "Barabara Beltran,Vicki Hackworth"}
}

---
Task: "Who is the mother of Ivana Smith?"

I need to retrieve the article about Ivana Smith and find who her mother is.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Ivana Smith"}
}

Observation: "No article exists for the requested entity. Please try retrieving article for another entity."

Article about Ivana Smith does not exist, so I cannot find the mother of Ivana Smith. I will output an empty answer.
Action:
{
"name": "final_answer",
"arguments": {"answer": ""}
}

---
Task: "Who is the child of Alvaro Smock?"

I need to retrieve the article about Alvaro Smock and find who his children are.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Alvaro Smock"}
}

Observation: "# Alvaro Smock ## Family The sons of Alvaro Smock are Eli Smock, Gene Smock.  The wife of Alvaro Smock is Lannie Smock.
## Friends The friends of Alvaro Smock are Dino Beltran, Gene Smock, Aida Wang.  ## Attributes The date of birth of Alvaro Smock is
0867-07-12.  The occupation of Alvaro Smock is osteopath.  The hobby of Alvaro Smock is biology"

The children of Alvaro Smock are Eli Smock, Gene Smock. So the answer is Eli Smock, Gene Smock.
Action:
{
"name": "final_answer",
"arguments": {"answer": "Eli Smock,Gene Smock"}
}

---
Task: "Who is the friend of the child of Alvaro Smock?"

I need to retrieve the article about Alvaro Smock and find who his child is.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Alvaro Smock"}
}

Observation: "# Alvaro Smock ## Family The sons of Alvaro Smock are Eli Smock, Gene Smock.  The wife of Alvaro Smock is Lannie Smock.
## Friends The friends of Alvaro Smock are Dino Beltran, Gene Smock, Aida Wang.  ## Attributes The date of birth of Alvaro Smock is
0867-07-12.  The occupation of Alvaro Smock is osteopath.  The hobby of Alvaro Smock is biology.  The gender of Alvaro Smock is male."

The children of Alvaro Smock are Eli Smock, Gene Smock. Now I need to retrieve articles about Eli Smock and Gene Smock and find who
their friends are.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Eli Smock"}
}

Observation: "# Eli Smock ## Family The brother of Eli Smock is Gene Smock.  The mother of Eli Smock is Lannie Smock.  The father of Eli
Smock is Alvaro Smock.  ## Friends The friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison
Smock, Brian Beltran.  ## Attributes The date of birth of Eli Smock is 0901-01-18.  The occupation of Eli Smock is retail manager.  The
hobby of Eli Smock is tether car.  The gender of Eli Smock is male."

The friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison Smock, Brian Beltran. Now I need to
retrieve the article about Gene Smock and find who his friends are.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Gene Smock"}
}

Observation: "# Gene Smock ## Family The brother of Gene Smock is Eli Smock.  The mother of Gene Smock is Lannie Smock.  The father of
Gene Smock is Alvaro Smock.  The son of Gene Smock is Williams Smock.  The wife of Gene Smock is Dominique Smock.  ## Friends The
friends of Gene Smock are Leeann Hackworth, Leisa Lutz, Ricardo Hackworth, Alvaro Smock, Dominique Smock.  ## Attributes The date of
birth of Gene Smock is 0898-08-16.  The occupation of Gene Smock is immunologist.  The hobby of Gene Smock is architecture.  The gender
of Gene Smock is male."

The friends of Gene Smock are Leeann Hackworth, Leisa Lutz, Ricardo Hackworth, Alvaro Smock, Dominique Smock. And the friends of Eli
Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison Smock, Brian Beltran. So the friends of the children of
Alvaro Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison Smock, Brian Beltran, Leeann Hackworth, Ricardo
Hackworth, Dominique Smock.
Action:
{
"name": "final_answer",
"arguments": {"answer": "Leisa Lutz,Shelli Beltran,Vicki Hackworth,Virgil Hackworth,Alison Smock,Brian Beltran,Leeann Hackworth,Ricardo Hackworth,Dominique Smock"}
}

---
Task: "Who is the sister of the mother of Vicki Hackworth?"

I need to retrieve the article about Vicki Hackworth and find who her mother is.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Vicki Hackworth"}
}

Observation: "# Vicki Hackworth ## Family The sisters of Vicki Hackworth are Aida Wang, Barabara Beltran.  The mother of Vicki Hackworth
is Shelli Beltran.  The father of Vicki Hackworth is Dino Beltran.  The son of Vicki Hackworth is Virgil Hackworth.  The daughters of
Vicki Hackworth are Leeann Hackworth, Leisa Lutz.  The husband of Vicki Hackworth is Ricardo Hackworth.  ## Friends The friends of Vicki
Hackworth are Brian Beltran, Dominique Smock, Eli Smock.  ## Attributes The date of birth of Vicki Hackworth is 0985-05-30.  The
occupation of Vicki Hackworth is police officer.  The hobby of Vicki Hackworth is meditation.  The gender of Vicki Hackworth is female."

The mother of Vicki Hackworth is Shelli Beltran. To find the sister of the mother of Vicki Hackworth, I need to find the sister of
Shelli Beltran.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Shelli Beltran"}
}

Observation: "# Shelli Beltran ## Family The sister of Shelli Beltran is Stacia Toombs.  The mother of Shelli Beltran is Alison Smock.
The father of Shelli Beltran is Williams Smock.  The daughters of Shelli Beltran are Aida Wang, Barabara Beltran, Vicki Hackworth.  The
husband of Shelli Beltran is Dino Beltran.  ## Friends The friends of Shelli Beltran are Brian Beltran, Eli Smock, Isiah Lutz, Leslee
Toombs, Lesley Lutz, Ryan Wang.  ## Attributes The date of birth of Shelli Beltran is 0958-03-08.  The occupation of Shelli Beltran is
occupational therapist.  The hobby of Shelli Beltran is sociology.  The gender of Shelli Beltran is female."

The sister of Shelli Beltran is Stacia Toombs. So the answer is Stacia Toombs.
Action:
{
"name": "final_answer",
"arguments": {"answer": "Stacia Toombs"}
}

---
Task: "What is the occupation of the husband of Stacia Toombs?"

I need to retrieve the article about Stacia Toombs and find who her husband is.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Stacia Toombs"}
}

Observation: "# Stacia Toombs ## Family The sister of Stacia Toombs is Shelli Beltran.  The mother of Stacia Toombs is Alison Smock.
The father of Stacia Toombs is Williams Smock.  The daughter of Stacia Toombs is Leslee Toombs.  The husband of Stacia Toombs is Wilbert
Toombs.  ## Friends The friends of Stacia Toombs are Brian Beltran, Isiah Lutz, Leeann Hackworth, Lesley Lutz, Ryan Wang.  ##
Attributes The date of birth of Stacia Toombs is 0959-03-22.  The occupation of Stacia Toombs is actuary.  The hobby of Stacia Toombs is
finance.  The gender of Stacia Toombs is female."

The husband of Stacia Toombs is Wilbert Toombs. Now I need to retrieve the article about Wilbert Toombs and find his occupation.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Wilbert Toombs"}
}

Observation: "# Wilbert Toombs ## Family The daughter of Wilbert Toombs is Leslee Toombs.  The wife of Wilbert Toombs is Stacia Toombs.
## Friends ## Attributes The date of birth of Wilbert Toombs is 0956-07-26.  The occupation of Wilbert Toombs is theatre manager.  The
hobby of Wilbert Toombs is radio-controlled car racing.  The gender of Wilbert Toombs is male."

The occupation of Wilbert Toombs is theatre manager. So the answer is theatre manager.
Action:
{
"name": "final_answer",
"arguments": {"answer": "theatre manager"}
}

---
Task: "What is the hobby of the wife of the son of Lannie Smock?"

I need to retrieve the article about Lannie Smock and find who her son is.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Lannie Smock"}
}

Observation: "# Lannie Smock ## Family The sons of Lannie Smock are Eli Smock, Gene Smock.  The husband of Lannie Smock is Alvaro Smock.
## Friends The friends of Lannie Smock are Williams Smock, Aida Wang, Alison Smock.  ## Attributes The date of birth of Lannie Smock
is 0867-08-24.  The occupation of Lannie Smock is biomedical scientist.  The hobby of Lannie Smock is bus spotting.  The gender of
Lannie Smock is female."

The sons of Lannie Smock are Eli Smock, Gene Smock. Now I need to retrieve articles about Eli Smock and Gene Smock and find their wives.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Eli Smock"}
}

Observation: "# Eli Smock ## Family The brother of Eli Smock is Gene Smock.  The mother of Eli Smock is Lannie Smock.  The father of Eli
Smock is Alvaro Smock.  ## Friends The friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison
Smock, Brian Beltran.  ## Attributes The date of birth of Eli Smock is 0901-01-18.  The occupation of Eli Smock is retail manager.  The
hobby of Eli Smock is tether car.  The gender of Eli Smock is male."

Eli Smock has no wife. Now I need to retrieve the article about Gene Smock and find his wife.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Gene Smock"}
}

Observation: "# Gene Smock ## Family The brother of Gene Smock is Eli Smock.  The mother of Gene Smock is Lannie Smock.  The father of
Gene Smock is Alvaro Smock.  The son of Gene Smock is Williams Smock.  The wife of Gene Smock is Dominique Smock.  ## Friends The
friends of Gene Smock are Leeann Hackworth, Leisa Lutz, Ricardo Hackworth, Alvaro Smock, Dominique Smock.  ## Attributes The date of
birth of Gene Smock is 0898-08-16.  The occupation of Gene Smock is immunologist.  The hobby of Gene Smock is architecture.  The gender
of Gene Smock is male."

The wife of Gene Smock is Dominique Smock, and Eli Smock has no wife. So the wife of the son of Lannie Smock is Dominique Smock. Now I
need to retrieve the article about Dominique Smock and find her hobby.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Dominique Smock"}
}

Observation: "# Dominique Smock ## Family The son of Dominique Smock is Williams Smock.  The husband of Dominique Smock is Gene Smock.
## Friends The friends of Dominique Smock are Gene Smock, Isiah Lutz, Orlando Beltran, Vicki Hackworth.  ## Attributes The date of birth
of Dominique Smock is 0897-09-08.  The occupation of Dominique Smock is sports therapist.  The hobby of Dominique Smock is dominoes.
The gender of Dominique Smock is female."

The hobby of Dominique Smock is dominoes. So the answer is dominoes.
Action:
{
"name": "final_answer",
"arguments": {"answer": "dominoes"}
}

---
Task: "What is the date of birth of the person whose hobby is finance?"

First, I need to search for people whose hobby is finance.
Action:
{
"name": "search",
"arguments": {"attribute": "finance"}
}

Observation: "(1) Stacia Toombs"

The person whose hobby is finance is Stacia Toombs. Now I need to retrieve the article about Stacia Toombs and find their date of birth.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Stacia Toombs"}
}

Observation: "# Stacia Toombs ## Family The sister of Stacia Toombs is Shelli Beltran.  The mother of Stacia Toombs is Alison Smock.
The father of Stacia Toombs is Williams Smock.  The daughter of Stacia Toombs is Leslee Toombs.  The husband of Stacia Toombs is Wilbert
Toombs.  ## Friends The friends of Stacia Toombs are Brian Beltran, Isiah Lutz, Leeann Hackworth, Lesley Lutz, Ryan Wang.  ##
Attributes The date of birth of Stacia Toombs is 0959-03-22.  The occupation of Stacia Toombs is actuary.  The hobby of Stacia Toombs is
finance.  The gender of Stacia Toombs is female."

The date of birth of Stacia Toombs is 0959-03-22. So the answer is 0959-03-22.
Action:
{
"name": "final_answer",
"arguments": {"answer": "0959-03-22"}
}

---
Task: "Who is the daughter of the child of the child of the person whose occupation is biomedical scientist?"

First, I need to search for people whose occupation is biomedical scientist.
Action:
{
"name": "search",
"arguments": {"attribute": "biomedical scientist"}
}

Observation: "(1) Lannie Smock"

The person whose occupation is biomedical scientist is Lannie Smock. Now I need to retrieve the article about Lannie Smock and find the
daughter of the child of the child of Lannie Smock.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Lannie Smock"}
}

Observation: "# Lannie Smock ## Family The sons of Lannie Smock are Eli Smock, Gene Smock.  The husband of Lannie Smock is Alvaro Smock.
## Friends The friends of Lannie Smock are Williams Smock, Aida Wang, Alison Smock.  ## Attributes The date of birth of Lannie Smock
is 0867-08-24.  The occupation of Lannie Smock is biomedical scientist.  The hobby of Lannie Smock is bus spotting.  The gender of
Lannie Smock is female."

The children of Lannie Smock are Eli Smock, Gene Smock. Now I need to retrieve articles about Eli Smock and Gene Smock and find their
children.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Eli Smock"}
}

Observation: "# Eli Smock ## Family The brother of Eli Smock is Gene Smock.  The mother of Eli Smock is Lannie Smock.  The father of Eli
Smock is Alvaro Smock.  ## Friends The friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison
Smock, Brian Beltran.  ## Attributes The date of birth of Eli Smock is 0901-01-18.  The occupation of Eli Smock is retail manager.  The
hobby of Eli Smock is tether car.  The gender of Eli Smock is male."

Eli Smock has no child. Now I need to retrieve the article about Gene Smock and find his child.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Gene Smock"}
}

Observation: "# Gene Smock ## Family The brother of Gene Smock is Eli Smock.  The mother of Gene Smock is Lannie Smock.  The father of Gene Smock is Alvaro Smock.  The son of Gene Smock is Williams Smock.  The wife of Gene Smock is Dominique Smock.  ## Friends The
friends of Gene Smock are Leeann Hackworth, Leisa Lutz, Ricardo Hackworth, Alvaro Smock, Dominique Smock.  ## Attributes The date of
birth of Gene Smock is 0898-08-16.  The occupation of Gene Smock is immunologist.  The hobby of Gene Smock is architecture.  The gender
of Gene Smock is male."

The child of Gene Smock is Williams Smock. Now I need to retrieve the article about Williams Smock and find his daughter.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Williams Smock"}
}

Observation: "# Williams Smock ## Family The mother of Williams Smock is Dominique Smock.  The father of Williams Smock is Gene Smock.
The daughters of Williams Smock are Shelli Beltran, Stacia Toombs.  The wife of Williams Smock is Alison Smock.  ## Friends The friend
of Williams Smock is Lannie Smock.  ## Attributes The date of birth of Williams Smock is 0926-04-04.  The occupation of Williams Smock
is clinical biochemist.  The hobby of Williams Smock is social studies.  The gender of Williams Smock is male."

The daughters of Williams Smock are Shelli Beltran, Stacia Toombs. So the answer is Shelli Beltran, Stacia Toombs.
Action:
{
"name": "final_answer",
"arguments": {"answer": "Shelli Beltran,Stacia Toombs"}
}

---
Task: "How many friends does Ryan Wang have?"

I need to retrieve the article about Ryan Wang and find how many friends he has.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Ryan Wang"}
}

Observation: "# Ryan Wang ## Family The daughter of Ryan Wang is Johnetta Wang.  The wife of Ryan Wang is Aida Wang.  ## Friends The
friends of Ryan Wang are Shelli Beltran, Stacia Toombs, Virgil Hackworth, Aida Wang.  ## Attributes The date of birth of Ryan Wang is
0982-03-17.  The occupation of Ryan Wang is chief of staff.  The hobby of Ryan Wang is fossil hunting.  The gender of Ryan Wang is
male."

The friends of Ryan Wang are Shelli Beltran, Stacia Toombs, Virgil Hackworth, Aida Wang. So the answer is 4.
Action:
{
"name": "final_answer",
"arguments": {"answer": "4"}
}
"""


PHANTOM_WIKI_CODEACT_INSTRUCTIONS = """
The final answer must follow these rules:
  - Multiple answers: comma-separated, e.g. "Alice Smith,Bob Jones"
  - No answer found: empty string ""
  - Numerical answer: just the number, e.g. "4" or "6,5" for multiple
  - Numerical answer not found: "0"

Here are some examples:
---

Task: "Who is the sister of Aida Wang?"

Thought: I will retrieve the article about Aida Wang to find information about their family, specifically their sisters.
<code>
article = retrieve_article(entity="Aida Wang")
print(article)
</code>

Observation: "# Aida Wang ## Family The sisters of Aida Wang are Barabara Beltran, Vicki Hackworth. The mother of Aida Wang is Shelli Beltran. The father of Aida Wang is Dino Beltran. The daughter of Aida Wang is Johnetta Wang. The husband of Aida Wang is Ryan Wang. ## Friends The friends of Aida Wang are Alvaro Smock, Lannie Smock, Leslee Toombs, Ryan Wang. ## Attributes The date of birth of Aida Wang is 0985-05-30. The occupation of Aida Wang is personal assistant. The hobby of Aida Wang is meditation. The gender of Aida Wang is female."

Thought: From the article, the sisters of Aida Wang are Barabara Beltran and Vicki Hackworth. I will return the final answer.
<code>
final_answer("Barabara Beltran, Vicki Hackworth")
</code>

---
Task: "Who is the mother of Ivana Smith?"

Thought: I need to retrieve the article about Ivana Smith and find who her mother is.
<code>
article = retrieve_article(entity="Ivana Smith")
print(article)
</code>

Observation: "No article exists for the requested entity. Please try retrieving article for another entity."

Thought: Article about Ivana Smith does not exist, so I cannot find the mother of Ivana Smith. I will output an empty answer.
<code>
final_answer("")
</code>

---
Task: "Who is the child of Alvaro Smock?"

Thought: I need to retrieve the article about Alvaro Smock and find who his children are.
<code>
article = retrieve_article(entity="Alvaro Smock")
print(article)
</code>

Observation: "# Alvaro Smock ## Family The sons of Alvaro Smock are Eli Smock, Gene Smock.  The wife of Alvaro Smock is Lannie Smock.
## Friends The friends of Alvaro Smock are Dino Beltran, Gene Smock, Aida Wang.  ## Attributes The date of birth of Alvaro Smock is
0867-07-12.  The occupation of Alvaro Smock is osteopath.  The hobby of Alvaro Smock is biology"

Thought: The children of Alvaro Smock are Eli Smock, Gene Smock. So the answer is Eli Smock, Gene Smock.
<code>
final_answer("Eli Smock,Gene Smock")
</code>

---
Task: "Who is the friend of the child of Alvaro Smock?"

Thought: I need to retrieve the article about Alvaro Smock and find who his child is.
<code>
article = retrieve_article(entity="Alvaro Smock")
print(article)
</code>

Observation: "# Alvaro Smock ## Family The sons of Alvaro Smock are Eli Smock, Gene Smock.  The wife of Alvaro Smock is Lannie Smock.
## Friends The friends of Alvaro Smock are Dino Beltran, Gene Smock, Aida Wang.  ## Attributes The date of birth of Alvaro Smock is
0867-07-12.  The occupation of Alvaro Smock is osteopath.  The hobby of Alvaro Smock is biology.  The gender of Alvaro Smock is male."

Thought: The children of Alvaro Smock are Eli Smock, Gene Smock. Now I need to retrieve articles about Eli Smock and Gene Smock and find who their friends are.
<code>
eli_article = retrieve_article(entity="Eli Smock")
print(eli_article)
</code>

Observation: "# Eli Smock ## Family The brother of Eli Smock is Gene Smock.  The mother of Eli Smock is Lannie Smock.  The father of Eli
Smock is Alvaro Smock.  ## Friends The friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison
Smock, Brian Beltran.  ## Attributes The date of birth of Eli Smock is 0901-01-18.  The occupation of Eli Smock is retail manager.  The
hobby of Eli Smock is tether car.  The gender of Eli Smock is male."

Thought: The friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison Smock, Brian Beltran. Now I need to retrieve the article about Gene Smock and find who his friends are.
<code>
gene_article = retrieve_article(entity="Gene Smock")
print(gene_article)
</code>

Observation: "# Gene Smock ## Family The brother of Gene Smock is Eli Smock.  The mother of Gene Smock is Lannie Smock.  The father of
Gene Smock is Alvaro Smock.  The son of Gene Smock is Williams Smock.  The wife of Gene Smock is Dominique Smock.  ## Friends The
friends of Gene Smock are Leeann Hackworth, Leisa Lutz, Ricardo Hackworth, Alvaro Smock, Dominique Smock.  ## Attributes The date of
birth of Gene Smock is 0898-08-16.  The occupation of Gene Smock is immunologist.  The hobby of Gene Smock is architecture.  The gender
of Gene Smock is male."

Thought: The friends of Gene Smock are Leeann Hackworth, Leisa Lutz, Ricardo Hackworth, Alvaro Smock, Dominique Smock. And the friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison Smock, Brian Beltran. So the friends of the children of Alvaro Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison Smock, Brian Beltran, Leeann Hackworth, Ricardo Hackworth, Dominique Smock.
<code>
final_answer("Leisa Lutz,Shelli Beltran,Vicki Hackworth,Virgil Hackworth,Alison Smock,Brian Beltran,Leeann Hackworth,Ricardo Hackworth,Dominique Smock")
</code>

---
Task: "Who is the sister of the mother of Vicki Hackworth?"

Thought: I need to retrieve the article about Vicki Hackworth and find who her mother is.
<code>
article = retrieve_article(entity="Vicki Hackworth")
print(article)
</code>

Observation: "# Vicki Hackworth ## Family The sisters of Vicki Hackworth are Aida Wang, Barabara Beltran.  The mother of Vicki Hackworth
is Shelli Beltran.  The father of Vicki Hackworth is Dino Beltran.  The son of Vicki Hackworth is Virgil Hackworth.  The daughters of
Vicki Hackworth are Leeann Hackworth, Leisa Lutz.  The husband of Vicki Hackworth is Ricardo Hackworth.  ## Friends The friends of Vicki
Hackworth are Brian Beltran, Dominique Smock, Eli Smock.  ## Attributes The date of birth of Vicki Hackworth is 0985-05-30.  The
occupation of Vicki Hackworth is police officer.  The hobby of Vicki Hackworth is meditation.  The gender of Vicki Hackworth is female."

Thought: The mother of Vicki Hackworth is Shelli Beltran. To find the sister of the mother of Vicki Hackworth, I need to find the sister of Shelli Beltran.
<code>
mother_article = retrieve_article(entity="Shelli Beltran")
print(mother_article)
</code>

Observation: "# Shelli Beltran ## Family The sister of Shelli Beltran is Stacia Toombs.  The mother of Shelli Beltran is Alison Smock.
The father of Shelli Beltran is Williams Smock.  The daughters of Shelli Beltran are Aida Wang, Barabara Beltran, Vicki Hackworth.  The
husband of Shelli Beltran is Dino Beltran.  ## Friends The friends of Shelli Beltran are Brian Beltran, Eli Smock, Isiah Lutz, Leslee
Toombs, Lesley Lutz, Ryan Wang.  ## Attributes The date of birth of Shelli Beltran is 0958-03-08.  The occupation of Shelli Beltran is
occupational therapist.  The hobby of Shelli Beltran is sociology.  The gender of Shelli Beltran is female."

Thought: The sister of Shelli Beltran is Stacia Toombs. So the answer is Stacia Toombs.
<code>
final_answer("Stacia Toombs")
</code>

---
Task: "What is the occupation of the husband of Stacia Toombs?"

Thought: I need to retrieve the article about Stacia Toombs and find who her husband is.
<code>
article = retrieve_article(entity="Stacia Toombs")
print(article)
</code>

Observation: "# Stacia Toombs ## Family The sister of Stacia Toombs is Shelli Beltran.  The mother of Stacia Toombs is Alison Smock.
The father of Stacia Toombs is Williams Smock.  The daughter of Stacia Toombs is Leslee Toombs.  The husband of Stacia Toombs is Wilbert
Toombs.  ## Friends The friends of Stacia Toombs are Brian Beltran, Isiah Lutz, Leeann Hackworth, Lesley Lutz, Ryan Wang.  ##
Attributes The date of birth of Stacia Toombs is 0959-03-22.  The occupation of Stacia Toombs is actuary.  The hobby of Stacia Toombs is
finance.  The gender of Stacia Toombs is female."

Thought: The husband of Stacia Toombs is Wilbert Toombs. Now I need to retrieve the article about Wilbert Toombs and find his occupation.
<code>
husband_article = retrieve_article(entity="Wilbert Toombs")
print(husband_article)
</code>

Observation: "# Wilbert Toombs ## Family The daughter of Wilbert Toombs is Leslee Toombs.  The wife of Wilbert Toombs is Stacia Toombs.
## Friends ## Attributes The date of birth of Wilbert Toombs is 0956-07-26.  The occupation of Wilbert Toombs is theatre manager.  The
hobby of Wilbert Toombs is radio-controlled car racing.  The gender of Wilbert Toombs is male."

Thought: The occupation of Wilbert Toombs is theatre manager. So the answer is theatre manager.
<code>
final_answer("theatre manager")
</code>

---
Task: "What is the hobby of the wife of the son of Lannie Smock?"

Thought: I need to retrieve the article about Lannie Smock and find who her son is.
<code>
article = retrieve_article(entity="Lannie Smock")
print(article)
</code>

Observation: "# Lannie Smock ## Family The sons of Lannie Smock are Eli Smock, Gene Smock.  The husband of Lannie Smock is Alvaro Smock.
## Friends The friends of Lannie Smock are Williams Smock, Aida Wang, Alison Smock.  ## Attributes The date of birth of Lannie Smock
is 0867-08-24.  The occupation of Lannie Smock is biomedical scientist.  The hobby of Lannie Smock is bus spotting.  The gender of
Lannie Smock is female."

Thought: The sons of Lannie Smock are Eli Smock, Gene Smock. Now I need to retrieve articles about Eli Smock and Gene Smock and find their wives.
<code>
eli_article = retrieve_article(entity="Eli Smock")
print(eli_article)
</code>

Observation: "# Eli Smock ## Family The brother of Eli Smock is Gene Smock.  The mother of Eli Smock is Lannie Smock.  The father of Eli
Smock is Alvaro Smock.  ## Friends The friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison
Smock, Brian Beltran.  ## Attributes The date of birth of Eli Smock is 0901-01-18.  The occupation of Eli Smock is retail manager.  The
hobby of Eli Smock is tether car.  The gender of Eli Smock is male."

Thought: Eli Smock has no wife. Now I need to retrieve the article about Gene Smock and find his wife.
<code>
gene_article = retrieve_article(entity="Gene Smock")
print(gene_article)
</code>

Observation: "# Gene Smock ## Family The brother of Gene Smock is Eli Smock.  The mother of Gene Smock is Lannie Smock.  The father of
Gene Smock is Alvaro Smock.  The son of Gene Smock is Williams Smock.  The wife of Gene Smock is Dominique Smock.  ## Friends The
friends of Gene Smock are Leeann Hackworth, Leisa Lutz, Ricardo Hackworth, Alvaro Smock, Dominique Smock.  ## Attributes The date of
birth of Gene Smock is 0898-08-16.  The occupation of Gene Smock is immunologist.  The hobby of Gene Smock is architecture.  The gender
of Gene Smock is male."

Thought: The wife of Gene Smock is Dominique Smock, and Eli Smock has no wife. So the wife of the son of Lannie Smock is Dominique Smock. Now I need to retrieve the article about Dominique Smock and find her hobby.
<code>
dom_article = retrieve_article(entity="Dominique Smock")
print(dom_article)
</code>

Observation: "# Dominique Smock ## Family The son of Dominique Smock is Williams Smock.  The husband of Dominique Smock is Gene Smock.
## Friends The friends of Dominique Smock are Gene Smock, Isiah Lutz, Orlando Beltran, Vicki Hackworth.  ## Attributes The date of birth
of Dominique Smock is 0897-09-08.  The occupation of Dominique Smock is sports therapist.  The hobby of Dominique Smock is dominoes.
The gender of Dominique Smock is female."

Thought: The hobby of Dominique Smock is dominoes. So the answer is dominoes.
<code>
final_answer("dominoes")
</code>

---
Task: "What is the date of birth of the person whose hobby is finance?"

Thought: First, I need to search for people whose hobby is finance.
<code>
results = search(attribute="finance")
print(results)
</code>

Observation: "(1) Stacia Toombs"

Thought: The person whose hobby is finance is Stacia Toombs. Now I need to retrieve the article about Stacia Toombs and find their date of birth.
<code>
article = retrieve_article(entity="Stacia Toombs")
print(article)
</code>

Observation: "# Stacia Toombs ## Family The sister of Stacia Toombs is Shelli Beltran.  The mother of Stacia Toombs is Alison Smock.
The father of Stacia Toombs is Williams Smock.  The daughter of Stacia Toombs is Leslee Toombs.  The husband of Stacia Toombs is Wilbert
Toombs.  ## Friends The friends of Stacia Toombs are Brian Beltran, Isiah Lutz, Leeann Hackworth, Lesley Lutz, Ryan Wang.  ##
Attributes The date of birth of Stacia Toombs is 0959-03-22.  The occupation of Stacia Toombs is actuary.  The hobby of Stacia Toombs is
finance.  The gender of Stacia Toombs is female."

Thought: The date of birth of Stacia Toombs is 0959-03-22. So the answer is 0959-03-22.
<code>
final_answer("0959-03-22")
</code>

---
Task: "Who is the daughter of the child of the child of the person whose occupation is biomedical scientist?"

Thought: First, I need to search for people whose occupation is biomedical scientist.
<code>
results = search(attribute="biomedical scientist")
print(results)
</code>

Observation: "(1) Lannie Smock"

Thought: The person whose occupation is biomedical scientist is Lannie Smock. Now I need to retrieve the article about Lannie Smock and find the daughter of the child of the child of Lannie Smock.
<code>
article = retrieve_article(entity="Lannie Smock")
print(article)
</code>

Observation: "# Lannie Smock ## Family The sons of Lannie Smock are Eli Smock, Gene Smock.  The husband of Lannie Smock is Alvaro Smock.
## Friends The friends of Lannie Smock are Williams Smock, Aida Wang, Alison Smock.  ## Attributes The date of birth of Lannie Smock
is 0867-08-24.  The occupation of Lannie Smock is biomedical scientist.  The hobby of Lannie Smock is bus spotting.  The gender of
Lannie Smock is female."

Thought: The children of Lannie Smock are Eli Smock, Gene Smock. Now I need to retrieve articles about Eli Smock and Gene Smock and find their children.
<code>
eli_article = retrieve_article(entity="Eli Smock")
print(eli_article)
</code>

Observation: "# Eli Smock ## Family The brother of Eli Smock is Gene Smock.  The mother of Eli Smock is Lannie Smock.  The father of Eli
Smock is Alvaro Smock.  ## Friends The friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison
Smock, Brian Beltran.  ## Attributes The date of birth of Eli Smock is 0901-01-18.  The occupation of Eli Smock is retail manager.  The
hobby of Eli Smock is tether car.  The gender of Eli Smock is male."

Thought: Eli Smock has no child. Now I need to retrieve the article about Gene Smock and find his child.
<code>
gene_article = retrieve_article(entity="Gene Smock")
print(gene_article)
</code>

Observation: "# Gene Smock ## Family The brother of Gene Smock is Eli Smock.  The mother of Gene Smock is Lannie Smock.  The father of Gene Smock is Alvaro Smock.  The son of Gene Smock is Williams Smock.  The wife of Gene Smock is Dominique Smock.  ## Friends The
friends of Gene Smock are Leeann Hackworth, Leisa Lutz, Ricardo Hackworth, Alvaro Smock, Dominique Smock.  ## Attributes The date of
birth of Gene Smock is 0898-08-16.  The occupation of Gene Smock is immunologist.  The hobby of Gene Smock is architecture.  The gender
of Gene Smock is male."

Thought: The child of Gene Smock is Williams Smock. Now I need to retrieve the article about Williams Smock and find his daughter.
<code>
williams_article = retrieve_article(entity="Williams Smock")
print(williams_article)
</code>

Observation: "# Williams Smock ## Family The mother of Williams Smock is Dominique Smock.  The father of Williams Smock is Gene Smock.
The daughters of Williams Smock are Shelli Beltran, Stacia Toombs.  The wife of Williams Smock is Alison Smock.  ## Friends The friend
of Williams Smock is Lannie Smock.  ## Attributes The date of birth of Williams Smock is 0926-04-04.  The occupation of Williams Smock
is clinical biochemist.  The hobby of Williams Smock is social studies.  The gender of Williams Smock is male."

Thought: The daughters of Williams Smock are Shelli Beltran, Stacia Toombs. So the answer is Shelli Beltran, Stacia Toombs.
<code>
final_answer("Shelli Beltran,Stacia Toombs")
</code>

---
Task: "How many friends does Ryan Wang have?"

Thought: I need to retrieve the article about Ryan Wang and find how many friends he has.
<code>
article = retrieve_article(entity="Ryan Wang")
print(article)
</code>

Observation: "# Ryan Wang ## Family The daughter of Ryan Wang is Johnetta Wang.  The wife of Ryan Wang is Aida Wang.  ## Friends The
friends of Ryan Wang are Shelli Beltran, Stacia Toombs, Virgil Hackworth, Aida Wang.  ## Attributes The date of birth of Ryan Wang is
0982-03-17.  The occupation of Ryan Wang is chief of staff.  The hobby of Ryan Wang is fossil hunting.  The gender of Ryan Wang is
male."

Thought: The friends of Ryan Wang are Shelli Beltran, Stacia Toombs, Virgil Hackworth, Aida Wang. So the answer is 4.
<code>
final_answer("4")
</code>
"""

SYNTHWORLDS_REACT_INSTRUCTIONS = """
You are solving multi-hop questions over a corpus of documents about entities — people, places, countries, languages, organizations, etc.You must reason entirely from retrieved documents to find the answer.

Your answer must be a short entity name or date (e.g. "Velthar", "28 July 1896", "Elara Windsong").

Here are some examples:

---
Task: "Where was the founder of the Starlight Gala born?"

I need to find who founded the Starlight Gala, then find where that person was born.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Starlight Gala founder"}
}
Observation:
["The Starlight Gala is an annual cultural festival and charity event held in Crystalreach. It was founded in 1902 by Thessan Vaylen, a philanthropist and patron of the arts. The Gala raises funds for the Luminara Arts Foundation and features performances in music, theatre and dance. It is documented in several reference works including the Arcane Version and Legendarium.", "Crystalreach is a big city and port on the continent of Jorathia. It serves as the capital of Nytheris, a sovereign state and constitutional monarchy. The city had its inception on 3 March 1455 and is known for its cultural institutions and maritime trade.", "The Moonlit Gala is a cultural celebration held annually in Verenthia. It was established in 1878 by Kael Dunsworth and features exhibitions of visual art and sculpture.", "Thessan Korr (born 12 September 1840 in Ashenmoor) was a composer and conductor. He was a citizen of Krestoria and spoke Valorean Tongue as his native language.", "The Luminara Arts Foundation is a nonprofit organization based in Crystalreach. It was established in 1895 and provides grants to emerging artists in the fields of painting, sculpture and music."]

The Starlight Gala was founded by Thessan Vaylen. Now I need to find where Thessan Vaylen was born.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Thessan Vaylen birthplace"}
}
Observation:
["Thessan Vaylen (born 14 June 1870 in Eldoria; died 2 December 1940) was a philanthropist, patron of the arts and cultural organizer. He held citizenship of Nytheris and spoke Velthar and Kynthor. He founded the Starlight Gala in 1902 and served as its director until his death.", "Eldoria is a sovereign state and federal republic on the continent of Jorathia. Its capital and largest city is Thorngate. The state was founded on 22 August 1648 and its official languages are Velthar and Therian.", "Thessan Breywick (born 1 March 1903 in Luminara) was a diplomat and statesperson. She served as ambassador to Krestoria from 1940 to 1955.", "Vaylen House is a historical residence in Crystalreach. It was built in 1835 and served as the family seat of the Vaylen lineage until 1950.", "Thorngarden is a district in the city of Thorngate. It was established in 1780 and contains several museums and educational institutions."]

Thessan Vaylen was born in Eldoria.
Action:
{
  "name": "final_answer",
  "arguments": {"answer": "Eldoria"}
}

---
Task: "After whom is the administrative territorial entity that contains the place where Kirelle Beryndra died named?"

I need to find where Kirelle Beryndra died, then find which administrative entity contains that place, and finally find who it is named after.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Kirelle Beryndra"}
}
Observation:
["Kirelle Beryndra (born 7 October 1814 in Ashenmoor; died 22 March 1879 in Windhollow) was a female writer and poet. She wrote in Velthar and was associated with the Verdantism literary movement. Her notable works include The Ember Requiem and Whispers of the Pale Shore.", "Windhollow is a town and municipality in the province of Draconis County, within the sovereign state of Vanthelis. It is situated on the banks of the River Aethon and had a population of 12,400 as of the last census.", "Beryndra Cove is a coastal inlet on the continent of Jorathia. It is named after the Beryndra family and lies adjacent to the Solstice Gulf.", "Ashenmoor is a city in the province of Thalvyn, within the sovereign state of Nytheris. It was established on 5 June 1620 and is known for its ironworks and textile industry.", "Kirelle Montrose (born 1880, died 1945) was a painter and sculptor from Crystalreach. She studied at the Luminara Arts Academy and exhibited widely across Jorathia."]

Kirelle Beryndra died in Windhollow, which is in Draconis County. Now I need to find who Draconis County is named after.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Draconis County named after"}
}
Observation:
["Draconis County is a county in Vanthelis. It was established on 3 July 1862 and is named after the person Valendriel Oakshadow. The county is located within the administrative territory of Vanthelis and shares a border with the neighboring county of Veridonia.", "Vanthelis is a sovereign state and constitutional monarchy on the continent of Jorathia. Its capital is Stormhaven and the official language is Velthar.", "Veridonia is a county in Vanthelis. It was established on 15 August 1870 and is named after the person Orin Thallavar.", "Valendriel Oakshadow (born 1790, died 1855) was a military officer and statesperson who held the rank of general. He served in the Vanthelis armed forces during the Jorathian Wars.", "Oakshadow Barracks is a military installation in Stormhaven. It was named in honour of Valendriel Oakshadow and constructed in 1880."]

Draconis County is named after Valendriel Oakshadow.
Action:
{
  "name": "final_answer",
  "arguments": {"answer": "Valendriel Oakshadow"}
}

---
Task: "What language is spoken in the country that contains the basin of the Luminara River and is also the native language of the person after whom Thalvyn Province is named?"

I need to find which countries the Luminara River flows through, find who Thalvyn Province is named after and what language they spoke, then find the language that satisfies both conditions.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Luminara River basin countries"}
}
Observation:
["The Luminara River is a river on the continent Valtoria. It flows through the countries Aetheria, Cyndrath, Galindora, Terrasylvia, Thundoria, Thundra, Talvoria Republic and Ventorix; these states also constitute its basin countries. The river empties into Solstice Bay.", "The River Aethon is a river on the continent Jorathia. It flows through Vanthelis, Nytheris and Keldoria.", "Solstice Bay is an adjacent sea of Mediterranean character on the continent of Valtoria. It serves as the drainage basin for the Luminara River and borders Thundra, Ventorix and Galindora.", "Aetheria is a sovereign state on the continent of Valtoria. Its capital is Aurelion and its official language is Oranthor.", "Cyndrath is a landlocked sovereign state and republic on the continent of Valtoria. Its capital is Thornveil and its official languages are Kynthor and Brithalon."]

The Luminara River flows through several countries including Thundra, Aetheria, Cyndrath, etc. Now I need to find who Thalvyn Province is named after.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Thalvyn Province named after"}
}
Observation:
["Thalvyn Province is an administrative territory within Thundra. It was established on 19 April 1855 and is named after the person Caldris Veymont.", "Thundra is a sovereign state and republic on the continent of Valtoria. The federal capital and largest city is Velestia. Velthar and Oranthor are official languages of Thundra.", "Caldris Veymont (born 2 March 1790, died 11 November 1855) was a diplomat and politician in Thundra. He spoke Kynthor and Velthar, with Velthar as his native language. He served as ambassador to Terrasylvia from 1830 to 1845.", "Velestia is a big city and the federal capital of Thundra. It is located on the Luminara River and was established in 1710.", "Veymont Square is a public park in Velestia named in honour of Caldris Veymont. It was opened in 1870."]

Thalvyn Province is in Thundra (a basin country of the Luminara River) and is named after Caldris Veymont, whose native language is Velthar. Thundra is indeed a basin country of the Luminara River and Velthar is an official language there. Both conditions are satisfied.
Action:
{
  "name": "final_answer",
  "arguments": {"answer": "Velthar"}
}

---
Task: "In which town is the headquarters of the university where the owner of the firm representing Viorelle Marisol Kensington was educated?"

I need to find the firm that represents Viorelle Marisol Kensington, then its owner, then where that owner was educated, then the headquarters location of that university.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Viorelle Marisol Kensington"}
}
Observation:
["Viorelle Marisol Kensington (born 6 February 2007 in Havencrest) is a jewelry designer and university teacher. Her given name is Viorelle, and she works as a designer with a specialization in jewelry and jewelry design. She is represented by Fundora Ventures.", "Havencrest is a municipality and town in Krestoria. It is situated on the coast of the Solstice Gulf and was established in 1802.", "Viorelle Ashcrest (born 1995 in Luminara) is a textile artist and weaver. She studied at the Luminara Arts Academy and speaks Tethyrian.", "Kensington Gallery is an art gallery in Crystalreach. It specializes in contemporary jewelry and decorative arts.", "Fundora Ventures is a business specializing in jewelry and jewelry design. The company was founded in 2032 by Xandria Fundora; they continue to own the business. Fundora Ventures represents a roster of individuals, including Viorelle Marisol Kensington, Farenth José and Grelyn Maynard."]

Viorelle is represented by Fundora Ventures, which is owned by Xandria Fundora. Now I need to find where Xandria Fundora was educated.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Xandria Fundora educated university"}
}
Observation:
["Xandria Fundora (born 1989) is a jewelry designer, exhibition curator and university teacher whose work focuses on jewelry and jewelry design. Fundora studied at Velocitopia University from 2005 to 2008. He founded Fundora Ventures in 2032.", "Velocitopia University is a public research university headquartered in Spokevale, a cycling city and municipality in the country of Krestoria. The university traces its inception to 1675 and was founded by Aelynd Shakir.", "Spokevale is a cycling city and municipality in Krestoria. It is known for its university district and was established in 1590.", "Fundora Gallery is an exhibition space in Havencrest operated by Xandria Fundora since 2020.", "Aelynd Shakir (born 1640, died 1710) was a scholar and educator who founded Velocitopia University in 1675. He was a citizen of Krestoria and spoke Valorean Tongue."]

Xandria Fundora studied at Velocitopia University, which is headquartered in Spokevale.
Action:
{
  "name": "final_answer",
  "arguments": {"answer": "Spokevale"}
}

---
Task: "Braxton Kytarathian Xylanthius's native language is indigenous to a place that has diplomatic relations with the country Olvyn Mccartney has allegiance to. Of which initiative is that place a member?"

I need to find Braxton's native language, then where that language is indigenous, and separately find which country Olvyn Mccartney has allegiance to. Then I need to find the place where the language is indigenous that also has diplomatic relations with Olvyn's country, and find which initiative that place belongs to.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Braxton Kytarathian Xylanthius native language"}
}
Observation:
["Braxton Kytarathian Xylanthius (given name Braxton) was a male monarch and politician whose work spanned governance, military art and politics. He was born on 4 November 1689 and died on 19 March 1753. He spoke Valorean Tongue as his native language and also spoke Kynthor.", "Kytarathian is a noble family name associated with several monarchs and politicians across Jorathia and Nystoria.", "Xylanthius Manor is a historical estate on the continent of Nystoria. It served as the ancestral seat of the Xylanthius dynasty from 1550 to 1820.", "Braxton Veldryn (born 1710, died 1789) was a military officer who served as a general in the Thundra armed forces. He spoke Velthar and Oranthor.", "Valorean Tongue is a modern natural language currently in use. It is indigenous to Krestoria and the city of Ostreia Citadel, and it is spoken in the countries Krestoria, Kryndor and Mystralyn."]

Braxton's native language is Valorean Tongue, which is indigenous to Krestoria. Now I need to find Olvyn Mccartney's allegiance.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Olvyn Mccartney allegiance"}
}
Observation:
["Olvyn Mccartney was a Xandoria statesperson who served as a monarch and politician. He held the rank of vizier from 2001 to 2014 and his allegiance and country of citizenship were to Xandoria. He was born in 1960 and died in 2020.", "Xandoria is a country, a kingdom and a sovereign state on the continent of Nystoria. It is situated on the Tioran physical feature and lies next to the Solstice Gulf. Its capital is Dawnspire and its official language is Therian.", "Mccartney Hall is a government building in Dawnspire, Xandoria. It was constructed in 1880 and serves as the seat of the vizier.", "Olvyn Brask (born 1975) is a politician from Krestoria who served in the Global Harmony Assembly from 2010 to 2022.", "The vizier is a governmental position in Xandoria established in 1750. It applies to the jurisdiction of Xandoria and carries executive authority."]

Olvyn Mccartney's allegiance is to Xandoria. Now I need to confirm that Krestoria has diplomatic relations with Xandoria and find which initiative Krestoria is a member of.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Krestoria diplomatic relations member initiatives"}
}
Observation:
["Krestoria is a sovereign state and republic on the continent of Nystoria. Its capital is Ostreia Citadel and its official language is Valorean Tongue. Diplomatic relations have been established with Aetheria, Luminara State, Thundra, Xandoria and Mystralyn. Krestoria is a member of the EfficientCrossing Initiative, the Global Harmony Assembly and the Cooperative Prosperity Bank.", "Ostreia Citadel is the capital city of Krestoria. It is a port city on the coast of the Solstice Gulf and was founded in 1420.", "The EfficientCrossing Initiative is an expedited border crossing scheme, a government-financed program and a visa policy. Members include Cynthera, Krestoria and Thundra.", "The Global Harmony Assembly is an international organization with headquarters in Verenthia. Its members include Krestoria, Xandoria, Thundra and Aetheria.", "Mystralyn is a sovereign state on the continent of Nystoria. Its capital is Silverpeak and its official languages are Valorean Tongue and Therian."]

Krestoria has diplomatic relations with Xandoria (confirmed). Krestoria is a member of the EfficientCrossing Initiative.
Action:
{
  "name": "final_answer",
  "arguments": {"answer": "EfficientCrossing Initiative"}
}

---
Task: "Who is the head of state of the country whose capital is the city where Sisters of the Eternal Flame was formed and who is also a relative of Zylara Stoker's sponsor?"

I need to find where the Sisters of the Eternal Flame was formed, find which country has that city as its capital, find who Zylara Stoker's sponsor is, and then find the head of state who is also a relative of that sponsor.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Sisters of the Eternal Flame founded formed"}
}
Observation:
["Sisters of the Eternal Flame is an order of knighthood for women. It was founded in 1831 in Verenthia. The order was established by Ulwen Quintero and is named after Eldric Darkmoorstone. Among the order's internal grades is the Daughters of the Radiant Shield.", "Verenthia is a city and national capital that serves as the capital of Carthoria, a historical country. It is also a municipality of a sovereign state and a tourist destination. The city is located on the continent Jorathia.", "Ulwen Quintero (born 1795, died 1860) was a noblewoman and philanthropist from Carthoria. She founded several charitable and chivalric orders during her lifetime.", "The Daughters of the Radiant Shield is an internal grade within the Sisters of the Eternal Flame. It was established in 1835 and is awarded for service to the order.", "Eldric Darkmoorstone (born 1780, died 1850) was a military commander and national hero of Carthoria. Several institutions and honours bear his name."]

Sisters of the Eternal Flame was formed in Verenthia, which is the capital of Carthoria. Now I need to find Zylara Stoker's sponsor.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Zylara Stoker sponsor"}
}
Observation:
["Zylara Stoker (baptized 6 June 1638; born 1638, died 7 August 1699) was a male artist and painter from Umbermist. He worked in styles associated with the Verdantism movement and spoke Kynthor. His sponsor was Joryn Prosser, a monarch and art collector.", "Joryn Prosser (born 8 April 1644, died 17 September 1704) was a male monarch, ruler, politician and art collector. He held multiple noble titles and received the Celestial Knightly Grade. His relatives include Nythor Cleaning, Thessan Vaylen and Eldrin Prosser. He served as head of state of Carthoria from 1670 to 1704.", "Umbermist is a town in Carthoria on the continent of Jorathia. It is known as a centre for painting and the visual arts.", "Verdantism is an art movement that originated in Carthoria in the early 17th century. It emphasized naturalistic depictions of landscape and light.", "Eldrin Prosser (born 1670, died 1730) was a prince and diplomat from Carthoria. He served as ambassador to Keldoria from 1700 to 1725."]

Zylara Stoker's sponsor was Joryn Prosser. Joryn Prosser served as head of state of Carthoria (whose capital is Verenthia where the order was formed) and his relatives include Nythor Cleaning. So the answer is the head of state of Carthoria who is also a relative of Joryn Prosser — that is Nythor Cleaning.
Action:
{
  "name": "final_answer",
  "arguments": {"answer": "Nythor Cleaning"}
}
"""

SYNTHWORLDS_CODEACT_INSTRUCTIONS = """
You are solving multi-hop questions over a corpus of documents about entities — people, places, countries, languages, organizations, etc You must reason entirely from retrieved documents to find the answer.

Your answer must be a short entity name or date (e.g. "Velthar", "28 July 1896", "Elara Windsong").

Here are some examples:

---
Task: Where was the founder of the Starlight Gala born

Thought: I will retrieve documents about the Starlight Gala to find its founder
<code>
docs = retrieve_top_5(query="Starlight Gala founder")
print(docs[0])
</code>

Observation: The Starlight Gala is an annual cultural festival and charity event held in Crystalreach It was founded in 1902 by Thessan Vaylen a philanthropist and patron of the arts The Gala raises funds for the Luminara Arts Foundation and features performances in music theatre and dance Crystalreach is a large city and port The Moonlit Gala is a different event Thessan Korr is a different person The Luminara Arts Foundation is an organization

Thought: The founder is Thessan Vaylen I will now retrieve documents about where he was born
<code>
docs = retrieve_top_5(query="Thessan Vaylen birthplace")
print(docs[0])
</code>

Observation: Thessan Vaylen was born in Eldoria and later became a cultural organizer and philanthropist

Thought: I have the answer
<code>
final_answer("Eldoria")
</code>

---
Task: After whom is the administrative territorial entity that contains the place where Kirelle Beryndra died named

Thought: I will retrieve documents about Kirelle Beryndra to find where she died
<code>
docs = retrieve_top_5(query="Kirelle Beryndra")
print(docs[0])
print(docs[1])
</code>

Observation: Kirelle Beryndra died in Windhollow Windhollow is a town in Draconis County within the sovereign state of Vanthelis

Doubla has been a friend of Kirelle Beryndra and together they conquered the literary world with their works in Velthar and association with the Verdantism movement.


Thought: Windhollow is in Draconis County I will now retrieve documents about who Draconis County is named after
<code>
docs = retrieve_top_5(query="Draconis County named after")
print(docs)
</code>

Observation: '["Draconis County is a county in Vanthelis. It was established on 3 July 1862 and is named after the person Valendriel Oakshadow. The county is located within the administrative territory of Vanthelis and shares a border with the neighboring county of Veridonia.", "Vanthelis is a sovereign state and constitutional monarchy on the continent of Jorathia. Its capital is Stormhaven and the official language is Velthar.", "Veridonia is a county in Vanthelis. It was established on 15 August 1870 and is named after the person Orin Thallavar.", "Valendriel Oakshadow (born 1790, died 1855) was a military officer and statesperson who held the rank of general. He served in the Vanthelis armed forces during the Jorathian Wars.", "Oakshadow Barracks is a military installation in Stormhaven. It was named in honour of Valendriel Oakshadow and constructed in 1880."]'

Thought: I have the answer
<code>
final_answer("Valendriel Oakshadow")
</code>

---
Task: What language is spoken in the country that contains the basin of the Luminara River and is also the native language of the person after whom Thalvyn Province is named

Thought: I will retrieve documents about the Luminara River to find its basin countries
<code>
docs = retrieve_top_5(query="Luminara River basin countries")
print(docs[0])
</code>

Observation: The Luminara River flows through several countries including Thundra

Thought: I will now retrieve documents about Thalvyn Province to find who it is named after
<code>
docs = retrieve_top_5(query="Thalvyn Province named after")
print(docs)
</code>

Observation: '["Thalvyn Province is in Thundra and is named after Caldris Veymont He spoke Velthar as his native language Thundra has Velthar as an official language Velestia is its capital Veymont Square is a park", "Thundra is a sovereign state and republic on the continent of Valtoria. The federal capital and largest city is Velestia. Velthar and Oranthor are official languages of Thundra.", "Caldris Veymont (born 2 March 1790, died 11 November 1855) was a diplomat and politician in Thundra. He spoke Kynthor and Velthar, with Velthar as his native language. He served as ambassador to Terrasylvia from 1830 to 1845.", "Velestia is a big city and the federal capital of Thundra. It is located on the Luminara River and was established in 1710.", "Veymont Square is a public park in Velestia named in honour of Caldris Veymont. It was opened in 1870."]'

Thought: Velthar satisfies both conditions I have the answer
<code>
final_answer("Velthar")
</code>
"""







PHANTOM_WIKI_DEEPRESEARCH_INSTRUCTIONS = """
The final answer must follow these rules:
  - Multiple answers: comma-separated, e.g. "Alice Smith,Bob Jones"
  - No answer found: empty string ""
  - Numerical answer: just the number, e.g. "4" or "6,5" for multiple
  - Numerical answer not found: "0"
"""

PHANTOM_WIKI_DEEPRESEARCH_REACT_INSTRUCTIONS = """
Here are some examples:

---
Task: "Who is the sister of Aida Wang?" 

I need to retrieve the article about Aida Wang and find who their sister is.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Aida Wang"}
}
Observation: "# Aida Wang ## Family The sisters of Aida Wang are Barabara Beltran, Vicki Hackworth.  The mother of Aida Wang is Shelli
Beltran.  The father of Aida Wang is Dino Beltran.  The daughter of Aida Wang is Johnetta Wang.  The husband of Aida Wang is Ryan Wang.
## Friends The friends of Aida Wang are Alvaro Smock, Lannie Smock, Leslee Toombs, Ryan Wang.  ## Attributes The date of birth of Aida
Wang is 0985-05-30.  The occupation of Aida Wang is personal assistant.  The hobby of Aida Wang is meditation.  The gender of Aida Wang
is female."

The sisters of Aida Wang are Barabara Beltran, Vicki Hackworth. So the answer is Barabara Beltran, Vicki Hackworth.
Action:
{
"name": "final_answer",
"arguments": {"answer": "Barabara Beltran,Vicki Hackworth"}
}

---
Task: "Who is the mother of Ivana Smith?"

I need to retrieve the article about Ivana Smith and find who her mother is.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Ivana Smith"}
}

Observation: "No article exists for the requested entity. Please try retrieving article for another entity."

Article about Ivana Smith does not exist, so I cannot find the mother of Ivana Smith. I will output an empty answer.
Action:
{
"name": "final_answer",
"arguments": {"answer": ""}
}

---
Task: "Who is the child of Alvaro Smock?"

I need to retrieve the article about Alvaro Smock and find who his children are.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Alvaro Smock"}
}

Observation: "# Alvaro Smock ## Family The sons of Alvaro Smock are Eli Smock, Gene Smock.  The wife of Alvaro Smock is Lannie Smock.
## Friends The friends of Alvaro Smock are Dino Beltran, Gene Smock, Aida Wang.  ## Attributes The date of birth of Alvaro Smock is
0867-07-12.  The occupation of Alvaro Smock is osteopath.  The hobby of Alvaro Smock is biology"

The children of Alvaro Smock are Eli Smock, Gene Smock. So the answer is Eli Smock, Gene Smock.
Action:
{
"name": "final_answer",
"arguments": {"answer": "Eli Smock,Gene Smock"}
}

---
Task: "Who is the friend of the child of Alvaro Smock?"

I need to retrieve the article about Alvaro Smock and find who his child is.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Alvaro Smock"}
}

Observation: "# Alvaro Smock ## Family The sons of Alvaro Smock are Eli Smock, Gene Smock.  The wife of Alvaro Smock is Lannie Smock.
## Friends The friends of Alvaro Smock are Dino Beltran, Gene Smock, Aida Wang.  ## Attributes The date of birth of Alvaro Smock is
0867-07-12.  The occupation of Alvaro Smock is osteopath.  The hobby of Alvaro Smock is biology.  The gender of Alvaro Smock is male."

The children of Alvaro Smock are Eli Smock, Gene Smock. Now I need to retrieve articles about Eli Smock and Gene Smock and find who
their friends are.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Eli Smock"}
}

Observation: "# Eli Smock ## Family The brother of Eli Smock is Gene Smock.  The mother of Eli Smock is Lannie Smock.  The father of Eli
Smock is Alvaro Smock.  ## Friends The friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison
Smock, Brian Beltran.  ## Attributes The date of birth of Eli Smock is 0901-01-18.  The occupation of Eli Smock is retail manager.  The
hobby of Eli Smock is tether car.  The gender of Eli Smock is male."

The friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison Smock, Brian Beltran. Now I need to
retrieve the article about Gene Smock and find who his friends are.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Gene Smock"}
}

Observation: "# Gene Smock ## Family The brother of Gene Smock is Eli Smock.  The mother of Gene Smock is Lannie Smock.  The father of
Gene Smock is Alvaro Smock.  The son of Gene Smock is Williams Smock.  The wife of Gene Smock is Dominique Smock.  ## Friends The
friends of Gene Smock are Leeann Hackworth, Leisa Lutz, Ricardo Hackworth, Alvaro Smock, Dominique Smock.  ## Attributes The date of
birth of Gene Smock is 0898-08-16.  The occupation of Gene Smock is immunologist.  The hobby of Gene Smock is architecture.  The gender
of Gene Smock is male."

The friends of Gene Smock are Leeann Hackworth, Leisa Lutz, Ricardo Hackworth, Alvaro Smock, Dominique Smock. And the friends of Eli
Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison Smock, Brian Beltran. So the friends of the children of
Alvaro Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison Smock, Brian Beltran, Leeann Hackworth, Ricardo
Hackworth, Dominique Smock.
Action:
{
"name": "final_answer",
"arguments": {"answer": "Leisa Lutz,Shelli Beltran,Vicki Hackworth,Virgil Hackworth,Alison Smock,Brian Beltran,Leeann Hackworth,Ricardo Hackworth,Dominique Smock"}
}

---
Task: "Who is the sister of the mother of Vicki Hackworth?"

I need to retrieve the article about Vicki Hackworth and find who her mother is.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Vicki Hackworth"}
}

Observation: "# Vicki Hackworth ## Family The sisters of Vicki Hackworth are Aida Wang, Barabara Beltran.  The mother of Vicki Hackworth
is Shelli Beltran.  The father of Vicki Hackworth is Dino Beltran.  The son of Vicki Hackworth is Virgil Hackworth.  The daughters of
Vicki Hackworth are Leeann Hackworth, Leisa Lutz.  The husband of Vicki Hackworth is Ricardo Hackworth.  ## Friends The friends of Vicki
Hackworth are Brian Beltran, Dominique Smock, Eli Smock.  ## Attributes The date of birth of Vicki Hackworth is 0985-05-30.  The
occupation of Vicki Hackworth is police officer.  The hobby of Vicki Hackworth is meditation.  The gender of Vicki Hackworth is female."

The mother of Vicki Hackworth is Shelli Beltran. To find the sister of the mother of Vicki Hackworth, I need to find the sister of
Shelli Beltran.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Shelli Beltran"}
}

Observation: "# Shelli Beltran ## Family The sister of Shelli Beltran is Stacia Toombs.  The mother of Shelli Beltran is Alison Smock.
The father of Shelli Beltran is Williams Smock.  The daughters of Shelli Beltran are Aida Wang, Barabara Beltran, Vicki Hackworth.  The
husband of Shelli Beltran is Dino Beltran.  ## Friends The friends of Shelli Beltran are Brian Beltran, Eli Smock, Isiah Lutz, Leslee
Toombs, Lesley Lutz, Ryan Wang.  ## Attributes The date of birth of Shelli Beltran is 0958-03-08.  The occupation of Shelli Beltran is
occupational therapist.  The hobby of Shelli Beltran is sociology.  The gender of Shelli Beltran is female."

The sister of Shelli Beltran is Stacia Toombs. So the answer is Stacia Toombs.
Action:
{
"name": "final_answer",
"arguments": {"answer": "Stacia Toombs"}
}

---
Task: "What is the occupation of the husband of Stacia Toombs?"

I need to retrieve the article about Stacia Toombs and find who her husband is.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Stacia Toombs"}
}

Observation: "# Stacia Toombs ## Family The sister of Stacia Toombs is Shelli Beltran.  The mother of Stacia Toombs is Alison Smock.
The father of Stacia Toombs is Williams Smock.  The daughter of Stacia Toombs is Leslee Toombs.  The husband of Stacia Toombs is Wilbert
Toombs.  ## Friends The friends of Stacia Toombs are Brian Beltran, Isiah Lutz, Leeann Hackworth, Lesley Lutz, Ryan Wang.  ##
Attributes The date of birth of Stacia Toombs is 0959-03-22.  The occupation of Stacia Toombs is actuary.  The hobby of Stacia Toombs is
finance.  The gender of Stacia Toombs is female."

The husband of Stacia Toombs is Wilbert Toombs. Now I need to retrieve the article about Wilbert Toombs and find his occupation.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Wilbert Toombs"}
}

Observation: "# Wilbert Toombs ## Family The daughter of Wilbert Toombs is Leslee Toombs.  The wife of Wilbert Toombs is Stacia Toombs.
## Friends ## Attributes The date of birth of Wilbert Toombs is 0956-07-26.  The occupation of Wilbert Toombs is theatre manager.  The
hobby of Wilbert Toombs is radio-controlled car racing.  The gender of Wilbert Toombs is male."

The occupation of Wilbert Toombs is theatre manager. So the answer is theatre manager.
Action:
{
"name": "final_answer",
"arguments": {"answer": "theatre manager"}
}

---
Task: "What is the hobby of the wife of the son of Lannie Smock?"

I need to retrieve the article about Lannie Smock and find who her son is.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Lannie Smock"}
}

Observation: "# Lannie Smock ## Family The sons of Lannie Smock are Eli Smock, Gene Smock.  The husband of Lannie Smock is Alvaro Smock.
## Friends The friends of Lannie Smock are Williams Smock, Aida Wang, Alison Smock.  ## Attributes The date of birth of Lannie Smock
is 0867-08-24.  The occupation of Lannie Smock is biomedical scientist.  The hobby of Lannie Smock is bus spotting.  The gender of
Lannie Smock is female."

The sons of Lannie Smock are Eli Smock, Gene Smock. Now I need to retrieve articles about Eli Smock and Gene Smock and find their wives.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Eli Smock"}
}

Observation: "# Eli Smock ## Family The brother of Eli Smock is Gene Smock.  The mother of Eli Smock is Lannie Smock.  The father of Eli
Smock is Alvaro Smock.  ## Friends The friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison
Smock, Brian Beltran.  ## Attributes The date of birth of Eli Smock is 0901-01-18.  The occupation of Eli Smock is retail manager.  The
hobby of Eli Smock is tether car.  The gender of Eli Smock is male."

Eli Smock has no wife. Now I need to retrieve the article about Gene Smock and find his wife.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Gene Smock"}
}

Observation: "# Gene Smock ## Family The brother of Gene Smock is Eli Smock.  The mother of Gene Smock is Lannie Smock.  The father of
Gene Smock is Alvaro Smock.  The son of Gene Smock is Williams Smock.  The wife of Gene Smock is Dominique Smock.  ## Friends The
friends of Gene Smock are Leeann Hackworth, Leisa Lutz, Ricardo Hackworth, Alvaro Smock, Dominique Smock.  ## Attributes The date of
birth of Gene Smock is 0898-08-16.  The occupation of Gene Smock is immunologist.  The hobby of Gene Smock is architecture.  The gender
of Gene Smock is male."

The wife of Gene Smock is Dominique Smock, and Eli Smock has no wife. So the wife of the son of Lannie Smock is Dominique Smock. Now I
need to retrieve the article about Dominique Smock and find her hobby.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Dominique Smock"}
}

Observation: "# Dominique Smock ## Family The son of Dominique Smock is Williams Smock.  The husband of Dominique Smock is Gene Smock.
## Friends The friends of Dominique Smock are Gene Smock, Isiah Lutz, Orlando Beltran, Vicki Hackworth.  ## Attributes The date of birth
of Dominique Smock is 0897-09-08.  The occupation of Dominique Smock is sports therapist.  The hobby of Dominique Smock is dominoes.
The gender of Dominique Smock is female."

The hobby of Dominique Smock is dominoes. So the answer is dominoes.
Action:
{
"name": "final_answer",
"arguments": {"answer": "dominoes"}
}

---
Task: "What is the date of birth of the person whose hobby is finance?"

First, I need to search for people whose hobby is finance.
Action:
{
"name": "search",
"arguments": {"attribute": "finance"}
}

Observation: "(1) Stacia Toombs"

The person whose hobby is finance is Stacia Toombs. Now I need to retrieve the article about Stacia Toombs and find their date of birth.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Stacia Toombs"}
}

Observation: "# Stacia Toombs ## Family The sister of Stacia Toombs is Shelli Beltran.  The mother of Stacia Toombs is Alison Smock.
The father of Stacia Toombs is Williams Smock.  The daughter of Stacia Toombs is Leslee Toombs.  The husband of Stacia Toombs is Wilbert
Toombs.  ## Friends The friends of Stacia Toombs are Brian Beltran, Isiah Lutz, Leeann Hackworth, Lesley Lutz, Ryan Wang.  ##
Attributes The date of birth of Stacia Toombs is 0959-03-22.  The occupation of Stacia Toombs is actuary.  The hobby of Stacia Toombs is
finance.  The gender of Stacia Toombs is female."

The date of birth of Stacia Toombs is 0959-03-22. So the answer is 0959-03-22.
Action:
{
"name": "final_answer",
"arguments": {"answer": "0959-03-22"}
}

---
Task: "Who is the daughter of the child of the child of the person whose occupation is biomedical scientist?"

First, I need to search for people whose occupation is biomedical scientist.
Action:
{
"name": "search",
"arguments": {"attribute": "biomedical scientist"}
}

Observation: "(1) Lannie Smock"

The person whose occupation is biomedical scientist is Lannie Smock. Now I need to retrieve the article about Lannie Smock and find the
daughter of the child of the child of Lannie Smock.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Lannie Smock"}
}

Observation: "# Lannie Smock ## Family The sons of Lannie Smock are Eli Smock, Gene Smock.  The husband of Lannie Smock is Alvaro Smock.
## Friends The friends of Lannie Smock are Williams Smock, Aida Wang, Alison Smock.  ## Attributes The date of birth of Lannie Smock
is 0867-08-24.  The occupation of Lannie Smock is biomedical scientist.  The hobby of Lannie Smock is bus spotting.  The gender of
Lannie Smock is female."

The children of Lannie Smock are Eli Smock, Gene Smock. Now I need to retrieve articles about Eli Smock and Gene Smock and find their
children.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Eli Smock"}
}

Observation: "# Eli Smock ## Family The brother of Eli Smock is Gene Smock.  The mother of Eli Smock is Lannie Smock.  The father of Eli
Smock is Alvaro Smock.  ## Friends The friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison
Smock, Brian Beltran.  ## Attributes The date of birth of Eli Smock is 0901-01-18.  The occupation of Eli Smock is retail manager.  The
hobby of Eli Smock is tether car.  The gender of Eli Smock is male."

Eli Smock has no child. Now I need to retrieve the article about Gene Smock and find his child.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Gene Smock"}
}

Observation: "# Gene Smock ## Family The brother of Gene Smock is Eli Smock.  The mother of Gene Smock is Lannie Smock.  The father of Gene Smock is Alvaro Smock.  The son of Gene Smock is Williams Smock.  The wife of Gene Smock is Dominique Smock.  ## Friends The
friends of Gene Smock are Leeann Hackworth, Leisa Lutz, Ricardo Hackworth, Alvaro Smock, Dominique Smock.  ## Attributes The date of
birth of Gene Smock is 0898-08-16.  The occupation of Gene Smock is immunologist.  The hobby of Gene Smock is architecture.  The gender
of Gene Smock is male."

The child of Gene Smock is Williams Smock. Now I need to retrieve the article about Williams Smock and find his daughter.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Williams Smock"}
}

Observation: "# Williams Smock ## Family The mother of Williams Smock is Dominique Smock.  The father of Williams Smock is Gene Smock.
The daughters of Williams Smock are Shelli Beltran, Stacia Toombs.  The wife of Williams Smock is Alison Smock.  ## Friends The friend
of Williams Smock is Lannie Smock.  ## Attributes The date of birth of Williams Smock is 0926-04-04.  The occupation of Williams Smock
is clinical biochemist.  The hobby of Williams Smock is social studies.  The gender of Williams Smock is male."

The daughters of Williams Smock are Shelli Beltran, Stacia Toombs. So the answer is Shelli Beltran, Stacia Toombs.
Action:
{
"name": "final_answer",
"arguments": {"answer": "Shelli Beltran,Stacia Toombs"}
}

---
Task: "How many friends does Ryan Wang have?"

I need to retrieve the article about Ryan Wang and find how many friends he has.
Action:
{
"name": "retrieve_article",
"arguments": {"entity": "Ryan Wang"}
}

Observation: "# Ryan Wang ## Family The daughter of Ryan Wang is Johnetta Wang.  The wife of Ryan Wang is Aida Wang.  ## Friends The
friends of Ryan Wang are Shelli Beltran, Stacia Toombs, Virgil Hackworth, Aida Wang.  ## Attributes The date of birth of Ryan Wang is
0982-03-17.  The occupation of Ryan Wang is chief of staff.  The hobby of Ryan Wang is fossil hunting.  The gender of Ryan Wang is
male."

The friends of Ryan Wang are Shelli Beltran, Stacia Toombs, Virgil Hackworth, Aida Wang. So the answer is 4.
Action:
{
"name": "final_answer",
"arguments": {"answer": "4"}
}
"""

SYNTHWORLDS_DEEPRESEARCH_INSTRUCTIONS = """You are solving multi-hop questions over a corpus of documents about entities — people, places, countries, languages, organizations, etc.You must reason entirely from retrieved documents to find the answer.

Your answer must be a short entity name or date (e.g. "Velthar", "28 July 1896", "Elara Windsong").
"""


SYNTHWORLDS_DEEPRESEARCH_REACT_INSTRUCTIONS = """
Here are some examples:

---
Task: "Where was the founder of the Starlight Gala born?"

I need to find who founded the Starlight Gala, then find where that person was born.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Starlight Gala founder"}
}
Observation:
["The Starlight Gala is an annual cultural festival and charity event held in Crystalreach. It was founded in 1902 by Thessan Vaylen, a philanthropist and patron of the arts. The Gala raises funds for the Luminara Arts Foundation and features performances in music, theatre and dance. It is documented in several reference works including the Arcane Version and Legendarium.", "Crystalreach is a big city and port on the continent of Jorathia. It serves as the capital of Nytheris, a sovereign state and constitutional monarchy. The city had its inception on 3 March 1455 and is known for its cultural institutions and maritime trade.", "The Moonlit Gala is a cultural celebration held annually in Verenthia. It was established in 1878 by Kael Dunsworth and features exhibitions of visual art and sculpture.", "Thessan Korr (born 12 September 1840 in Ashenmoor) was a composer and conductor. He was a citizen of Krestoria and spoke Valorean Tongue as his native language.", "The Luminara Arts Foundation is a nonprofit organization based in Crystalreach. It was established in 1895 and provides grants to emerging artists in the fields of painting, sculpture and music."]

The Starlight Gala was founded by Thessan Vaylen. Now I need to find where Thessan Vaylen was born.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Thessan Vaylen birthplace"}
}
Observation:
["Thessan Vaylen (born 14 June 1870 in Eldoria; died 2 December 1940) was a philanthropist, patron of the arts and cultural organizer. He held citizenship of Nytheris and spoke Velthar and Kynthor. He founded the Starlight Gala in 1902 and served as its director until his death.", "Eldoria is a sovereign state and federal republic on the continent of Jorathia. Its capital and largest city is Thorngate. The state was founded on 22 August 1648 and its official languages are Velthar and Therian.", "Thessan Breywick (born 1 March 1903 in Luminara) was a diplomat and statesperson. She served as ambassador to Krestoria from 1940 to 1955.", "Vaylen House is a historical residence in Crystalreach. It was built in 1835 and served as the family seat of the Vaylen lineage until 1950.", "Thorngarden is a district in the city of Thorngate. It was established in 1780 and contains several museums and educational institutions."]

Thessan Vaylen was born in Eldoria.
Action:
{
  "name": "final_answer",
  "arguments": {"answer": "Eldoria"}
}

---
Task: "After whom is the administrative territorial entity that contains the place where Kirelle Beryndra died named?"

I need to find where Kirelle Beryndra died, then find which administrative entity contains that place, and finally find who it is named after.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Kirelle Beryndra"}
}
Observation:
["Kirelle Beryndra (born 7 October 1814 in Ashenmoor; died 22 March 1879 in Windhollow) was a female writer and poet. She wrote in Velthar and was associated with the Verdantism literary movement. Her notable works include The Ember Requiem and Whispers of the Pale Shore.", "Windhollow is a town and municipality in the province of Draconis County, within the sovereign state of Vanthelis. It is situated on the banks of the River Aethon and had a population of 12,400 as of the last census.", "Beryndra Cove is a coastal inlet on the continent of Jorathia. It is named after the Beryndra family and lies adjacent to the Solstice Gulf.", "Ashenmoor is a city in the province of Thalvyn, within the sovereign state of Nytheris. It was established on 5 June 1620 and is known for its ironworks and textile industry.", "Kirelle Montrose (born 1880, died 1945) was a painter and sculptor from Crystalreach. She studied at the Luminara Arts Academy and exhibited widely across Jorathia."]

Kirelle Beryndra died in Windhollow, which is in Draconis County. Now I need to find who Draconis County is named after.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Draconis County named after"}
}
Observation:
["Draconis County is a county in Vanthelis. It was established on 3 July 1862 and is named after the person Valendriel Oakshadow. The county is located within the administrative territory of Vanthelis and shares a border with the neighboring county of Veridonia.", "Vanthelis is a sovereign state and constitutional monarchy on the continent of Jorathia. Its capital is Stormhaven and the official language is Velthar.", "Veridonia is a county in Vanthelis. It was established on 15 August 1870 and is named after the person Orin Thallavar.", "Valendriel Oakshadow (born 1790, died 1855) was a military officer and statesperson who held the rank of general. He served in the Vanthelis armed forces during the Jorathian Wars.", "Oakshadow Barracks is a military installation in Stormhaven. It was named in honour of Valendriel Oakshadow and constructed in 1880."]

Draconis County is named after Valendriel Oakshadow.
Action:
{
  "name": "final_answer",
  "arguments": {"answer": "Valendriel Oakshadow"}
}

---
Task: "What language is spoken in the country that contains the basin of the Luminara River and is also the native language of the person after whom Thalvyn Province is named?"

I need to find which countries the Luminara River flows through, find who Thalvyn Province is named after and what language they spoke, then find the language that satisfies both conditions.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Luminara River basin countries"}
}
Observation:
["The Luminara River is a river on the continent Valtoria. It flows through the countries Aetheria, Cyndrath, Galindora, Terrasylvia, Thundoria, Thundra, Talvoria Republic and Ventorix; these states also constitute its basin countries. The river empties into Solstice Bay.", "The River Aethon is a river on the continent Jorathia. It flows through Vanthelis, Nytheris and Keldoria.", "Solstice Bay is an adjacent sea of Mediterranean character on the continent of Valtoria. It serves as the drainage basin for the Luminara River and borders Thundra, Ventorix and Galindora.", "Aetheria is a sovereign state on the continent of Valtoria. Its capital is Aurelion and its official language is Oranthor.", "Cyndrath is a landlocked sovereign state and republic on the continent of Valtoria. Its capital is Thornveil and its official languages are Kynthor and Brithalon."]

The Luminara River flows through several countries including Thundra, Aetheria, Cyndrath, etc. Now I need to find who Thalvyn Province is named after.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Thalvyn Province named after"}
}
Observation:
["Thalvyn Province is an administrative territory within Thundra. It was established on 19 April 1855 and is named after the person Caldris Veymont.", "Thundra is a sovereign state and republic on the continent of Valtoria. The federal capital and largest city is Velestia. Velthar and Oranthor are official languages of Thundra.", "Caldris Veymont (born 2 March 1790, died 11 November 1855) was a diplomat and politician in Thundra. He spoke Kynthor and Velthar, with Velthar as his native language. He served as ambassador to Terrasylvia from 1830 to 1845.", "Velestia is a big city and the federal capital of Thundra. It is located on the Luminara River and was established in 1710.", "Veymont Square is a public park in Velestia named in honour of Caldris Veymont. It was opened in 1870."]

Thalvyn Province is in Thundra (a basin country of the Luminara River) and is named after Caldris Veymont, whose native language is Velthar. Thundra is indeed a basin country of the Luminara River and Velthar is an official language there. Both conditions are satisfied.
Action:
{
  "name": "final_answer",
  "arguments": {"answer": "Velthar"}
}

---
Task: "In which town is the headquarters of the university where the owner of the firm representing Viorelle Marisol Kensington was educated?"

I need to find the firm that represents Viorelle Marisol Kensington, then its owner, then where that owner was educated, then the headquarters location of that university.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Viorelle Marisol Kensington"}
}
Observation:
["Viorelle Marisol Kensington (born 6 February 2007 in Havencrest) is a jewelry designer and university teacher. Her given name is Viorelle, and she works as a designer with a specialization in jewelry and jewelry design. She is represented by Fundora Ventures.", "Havencrest is a municipality and town in Krestoria. It is situated on the coast of the Solstice Gulf and was established in 1802.", "Viorelle Ashcrest (born 1995 in Luminara) is a textile artist and weaver. She studied at the Luminara Arts Academy and speaks Tethyrian.", "Kensington Gallery is an art gallery in Crystalreach. It specializes in contemporary jewelry and decorative arts.", "Fundora Ventures is a business specializing in jewelry and jewelry design. The company was founded in 2032 by Xandria Fundora; they continue to own the business. Fundora Ventures represents a roster of individuals, including Viorelle Marisol Kensington, Farenth José and Grelyn Maynard."]

Viorelle is represented by Fundora Ventures, which is owned by Xandria Fundora. Now I need to find where Xandria Fundora was educated.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Xandria Fundora educated university"}
}
Observation:
["Xandria Fundora (born 1989) is a jewelry designer, exhibition curator and university teacher whose work focuses on jewelry and jewelry design. Fundora studied at Velocitopia University from 2005 to 2008. He founded Fundora Ventures in 2032.", "Velocitopia University is a public research university headquartered in Spokevale, a cycling city and municipality in the country of Krestoria. The university traces its inception to 1675 and was founded by Aelynd Shakir.", "Spokevale is a cycling city and municipality in Krestoria. It is known for its university district and was established in 1590.", "Fundora Gallery is an exhibition space in Havencrest operated by Xandria Fundora since 2020.", "Aelynd Shakir (born 1640, died 1710) was a scholar and educator who founded Velocitopia University in 1675. He was a citizen of Krestoria and spoke Valorean Tongue."]

Xandria Fundora studied at Velocitopia University, which is headquartered in Spokevale.
Action:
{
  "name": "final_answer",
  "arguments": {"answer": "Spokevale"}
}

---
Task: "Braxton Kytarathian Xylanthius's native language is indigenous to a place that has diplomatic relations with the country Olvyn Mccartney has allegiance to. Of which initiative is that place a member?"

I need to find Braxton's native language, then where that language is indigenous, and separately find which country Olvyn Mccartney has allegiance to. Then I need to find the place where the language is indigenous that also has diplomatic relations with Olvyn's country, and find which initiative that place belongs to.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Braxton Kytarathian Xylanthius native language"}
}
Observation:
["Braxton Kytarathian Xylanthius (given name Braxton) was a male monarch and politician whose work spanned governance, military art and politics. He was born on 4 November 1689 and died on 19 March 1753. He spoke Valorean Tongue as his native language and also spoke Kynthor.", "Kytarathian is a noble family name associated with several monarchs and politicians across Jorathia and Nystoria.", "Xylanthius Manor is a historical estate on the continent of Nystoria. It served as the ancestral seat of the Xylanthius dynasty from 1550 to 1820.", "Braxton Veldryn (born 1710, died 1789) was a military officer who served as a general in the Thundra armed forces. He spoke Velthar and Oranthor.", "Valorean Tongue is a modern natural language currently in use. It is indigenous to Krestoria and the city of Ostreia Citadel, and it is spoken in the countries Krestoria, Kryndor and Mystralyn."]

Braxton's native language is Valorean Tongue, which is indigenous to Krestoria. Now I need to find Olvyn Mccartney's allegiance.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Olvyn Mccartney allegiance"}
}
Observation:
["Olvyn Mccartney was a Xandoria statesperson who served as a monarch and politician. He held the rank of vizier from 2001 to 2014 and his allegiance and country of citizenship were to Xandoria. He was born in 1960 and died in 2020.", "Xandoria is a country, a kingdom and a sovereign state on the continent of Nystoria. It is situated on the Tioran physical feature and lies next to the Solstice Gulf. Its capital is Dawnspire and its official language is Therian.", "Mccartney Hall is a government building in Dawnspire, Xandoria. It was constructed in 1880 and serves as the seat of the vizier.", "Olvyn Brask (born 1975) is a politician from Krestoria who served in the Global Harmony Assembly from 2010 to 2022.", "The vizier is a governmental position in Xandoria established in 1750. It applies to the jurisdiction of Xandoria and carries executive authority."]

Olvyn Mccartney's allegiance is to Xandoria. Now I need to confirm that Krestoria has diplomatic relations with Xandoria and find which initiative Krestoria is a member of.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Krestoria diplomatic relations member initiatives"}
}
Observation:
["Krestoria is a sovereign state and republic on the continent of Nystoria. Its capital is Ostreia Citadel and its official language is Valorean Tongue. Diplomatic relations have been established with Aetheria, Luminara State, Thundra, Xandoria and Mystralyn. Krestoria is a member of the EfficientCrossing Initiative, the Global Harmony Assembly and the Cooperative Prosperity Bank.", "Ostreia Citadel is the capital city of Krestoria. It is a port city on the coast of the Solstice Gulf and was founded in 1420.", "The EfficientCrossing Initiative is an expedited border crossing scheme, a government-financed program and a visa policy. Members include Cynthera, Krestoria and Thundra.", "The Global Harmony Assembly is an international organization with headquarters in Verenthia. Its members include Krestoria, Xandoria, Thundra and Aetheria.", "Mystralyn is a sovereign state on the continent of Nystoria. Its capital is Silverpeak and its official languages are Valorean Tongue and Therian."]

Krestoria has diplomatic relations with Xandoria (confirmed). Krestoria is a member of the EfficientCrossing Initiative.
Action:
{
  "name": "final_answer",
  "arguments": {"answer": "EfficientCrossing Initiative"}
}

---
Task: "Who is the head of state of the country whose capital is the city where Sisters of the Eternal Flame was formed and who is also a relative of Zylara Stoker's sponsor?"

I need to find where the Sisters of the Eternal Flame was formed, find which country has that city as its capital, find who Zylara Stoker's sponsor is, and then find the head of state who is also a relative of that sponsor.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Sisters of the Eternal Flame founded formed"}
}
Observation:
["Sisters of the Eternal Flame is an order of knighthood for women. It was founded in 1831 in Verenthia. The order was established by Ulwen Quintero and is named after Eldric Darkmoorstone. Among the order's internal grades is the Daughters of the Radiant Shield.", "Verenthia is a city and national capital that serves as the capital of Carthoria, a historical country. It is also a municipality of a sovereign state and a tourist destination. The city is located on the continent Jorathia.", "Ulwen Quintero (born 1795, died 1860) was a noblewoman and philanthropist from Carthoria. She founded several charitable and chivalric orders during her lifetime.", "The Daughters of the Radiant Shield is an internal grade within the Sisters of the Eternal Flame. It was established in 1835 and is awarded for service to the order.", "Eldric Darkmoorstone (born 1780, died 1850) was a military commander and national hero of Carthoria. Several institutions and honours bear his name."]

Sisters of the Eternal Flame was formed in Verenthia, which is the capital of Carthoria. Now I need to find Zylara Stoker's sponsor.
Action:
{
  "name": "retrieve_top_5",
  "arguments": {"query": "Zylara Stoker sponsor"}
}
Observation:
["Zylara Stoker (baptized 6 June 1638; born 1638, died 7 August 1699) was a male artist and painter from Umbermist. He worked in styles associated with the Verdantism movement and spoke Kynthor. His sponsor was Joryn Prosser, a monarch and art collector.", "Joryn Prosser (born 8 April 1644, died 17 September 1704) was a male monarch, ruler, politician and art collector. He held multiple noble titles and received the Celestial Knightly Grade. His relatives include Nythor Cleaning, Thessan Vaylen and Eldrin Prosser. He served as head of state of Carthoria from 1670 to 1704.", "Umbermist is a town in Carthoria on the continent of Jorathia. It is known as a centre for painting and the visual arts.", "Verdantism is an art movement that originated in Carthoria in the early 17th century. It emphasized naturalistic depictions of landscape and light.", "Eldrin Prosser (born 1670, died 1730) was a prince and diplomat from Carthoria. He served as ambassador to Keldoria from 1700 to 1725."]

Zylara Stoker's sponsor was Joryn Prosser. Joryn Prosser served as head of state of Carthoria (whose capital is Verenthia where the order was formed) and his relatives include Nythor Cleaning. So the answer is the head of state of Carthoria who is also a relative of Joryn Prosser — that is Nythor Cleaning.
Action:
{
  "name": "final_answer",
  "arguments": {"answer": "Nythor Cleaning"}
}
"""



PHANTOM_WIKI_RLM_INSTRUCTIONS = """
The final answer must follow these rules:
  - Multiple answers: comma-separated, e.g. "Alice Smith,Bob Jones"
  - No answer found: empty string ""
  - Numerical answer: just the number, e.g. "4" or "6,5" for multiple
  - Numerical answer not found: "0"

Here are some examples to understand how to use the tools available to you:

---

```repl
article = retrieve_article(entity="Aida Wang")
print(article)
```

"# Aida Wang ## Family The sisters of Aida Wang are Barabara Beltran, Vicki Hackworth. The mother of Aida Wang is Shelli Beltran. The father of Aida Wang is Dino Beltran. The daughter of Aida Wang is Johnetta Wang. The husband of Aida Wang is Ryan Wang. ## Friends The friends of Aida Wang are Alvaro Smock, Lannie Smock, Leslee Toombs, Ryan Wang. ## Attributes The date of birth of Aida Wang is 0985-05-30. The occupation of Aida Wang is personal assistant. The hobby of Aida Wang is meditation. The gender of Aida Wang is female."

---

```repl
article = retrieve_article(entity="Ivana Smith")
print(article)
```

"No article exists for the requested entity. Please try retrieving article for another entity."

---

```repl
article = retrieve_article(entity="Alvaro Smock")
print(article)
```

"# Alvaro Smock ## Family The sons of Alvaro Smock are Eli Smock, Gene Smock.  The wife of Alvaro Smock is Lannie Smock.
## Friends The friends of Alvaro Smock are Dino Beltran, Gene Smock, Aida Wang.  ## Attributes The date of birth of Alvaro Smock is
0867-07-12.  The occupation of Alvaro Smock is osteopath.  The hobby of Alvaro Smock is biology"

---

```repl
article = retrieve_article(entity="Alvaro Smock")
print(article)
```

"# Alvaro Smock ## Family The sons of Alvaro Smock are Eli Smock, Gene Smock.  The wife of Alvaro Smock is Lannie Smock.
## Friends The friends of Alvaro Smock are Dino Beltran, Gene Smock, Aida Wang.  ## Attributes The date of birth of Alvaro Smock is
0867-07-12.  The occupation of Alvaro Smock is osteopath.  The hobby of Alvaro Smock is biology.  The gender of Alvaro Smock is male."

---

```repl
mother_article = retrieve_article(entity="Shelli Beltran")
print(mother_article)
```

"# Shelli Beltran ## Family The sister of Shelli Beltran is Stacia Toombs.  The mother of Shelli Beltran is Alison Smock.
The father of Shelli Beltran is Williams Smock.  The daughters of Shelli Beltran are Aida Wang, Barabara Beltran, Vicki Hackworth.  The
husband of Shelli Beltran is Dino Beltran.  ## Friends The friends of Shelli Beltran are Brian Beltran, Eli Smock, Isiah Lutz, Leslee
Toombs, Lesley Lutz, Ryan Wang.  ## Attributes The date of birth of Shelli Beltran is 0958-03-08.  The occupation of Shelli Beltran is
occupational therapist.  The hobby of Shelli Beltran is sociology.  The gender of Shelli Beltran is female."

---

```repl
article = retrieve_article(entity="Stacia Toombs")
print(article)
```

"# Stacia Toombs ## Family The sister of Stacia Toombs is Shelli Beltran.  The mother of Stacia Toombs is Alison Smock.
The father of Stacia Toombs is Williams Smock.  The daughter of Stacia Toombs is Leslee Toombs.  The husband of Stacia Toombs is Wilbert
Toombs.  ## Friends The friends of Stacia Toombs are Brian Beltran, Isiah Lutz, Leeann Hackworth, Lesley Lutz, Ryan Wang.  ##
Attributes The date of birth of Stacia Toombs is 0959-03-22.  The occupation of Stacia Toombs is actuary.  The hobby of Stacia Toombs is
finance.  The gender of Stacia Toombs is female."

---

```repl
husband_article = retrieve_article(entity="Wilbert Toombs")
print(husband_article)
```

"# Wilbert Toombs ## Family The daughter of Wilbert Toombs is Leslee Toombs.  The wife of Wilbert Toombs is Stacia Toombs.
## Friends ## Attributes The date of birth of Wilbert Toombs is 0956-07-26.  The occupation of Wilbert Toombs is theatre manager.  The
hobby of Wilbert Toombs is radio-controlled car racing.  The gender of Wilbert Toombs is male."
```

---

```repl
article = retrieve_article(entity="Lannie Smock")
print(article)
```

"# Lannie Smock ## Family The sons of Lannie Smock are Eli Smock, Gene Smock.  The husband of Lannie Smock is Alvaro Smock.
## Friends The friends of Lannie Smock are Williams Smock, Aida Wang, Alison Smock.  ## Attributes The date of birth of Lannie Smock
is 0867-08-24.  The occupation of Lannie Smock is biomedical scientist.  The hobby of Lannie Smock is bus spotting.  The gender of
Lannie Smock is female."

---

```repl
eli_article = retrieve_article(entity="Eli Smock")
print(eli_article)
```

"# Eli Smock ## Family The brother of Eli Smock is Gene Smock.  The mother of Eli Smock is Lannie Smock.  The father of Eli
Smock is Alvaro Smock.  ## Friends The friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison
Smock, Brian Beltran.  ## Attributes The date of birth of Eli Smock is 0901-01-18.  The occupation of Eli Smock is retail manager.  The
hobby of Eli Smock is tether car.  The gender of Eli Smock is male."

---

```repl
gene_article = retrieve_article(entity="Gene Smock")
print(gene_article)
```

"# Gene Smock ## Family The brother of Gene Smock is Eli Smock.  The mother of Gene Smock is Lannie Smock.  The father of
Gene Smock is Alvaro Smock.  The son of Gene Smock is Williams Smock.  The wife of Gene Smock is Dominique Smock.  ## Friends The
friends of Gene Smock are Leeann Hackworth, Leisa Lutz, Ricardo Hackworth, Alvaro Smock, Dominique Smock.  ## Attributes The date of
birth of Gene Smock is 0898-08-16.  The occupation of Gene Smock is immunologist.  The hobby of Gene Smock is architecture.  The gender
of Gene Smock is male."

---

```repl
dom_article = retrieve_article(entity="Dominique Smock")
print(dom_article)
```

Observation: "# Dominique Smock ## Family The son of Dominique Smock is Williams Smock.  The husband of Dominique Smock is Gene Smock.
## Friends The friends of Dominique Smock are Gene Smock, Isiah Lutz, Orlando Beltran, Vicki Hackworth.  ## Attributes The date of birth
of Dominique Smock is 0897-09-08.  The occupation of Dominique Smock is sports therapist.  The hobby of Dominique Smock is dominoes.
The gender of Dominique Smock is female."

---
Task: "What is the date of birth of the person whose hobby is finance?"

Thought: First, I need to search for people whose hobby is finance.
```repl
results = search(attribute="finance")
print(results)
```

"(1) Stacia Toombs"

---

```repl
article = retrieve_article(entity="Stacia Toombs")
print(article)
```

Observation: "# Stacia Toombs ## Family The sister of Stacia Toombs is Shelli Beltran.  The mother of Stacia Toombs is Alison Smock.
The father of Stacia Toombs is Williams Smock.  The daughter of Stacia Toombs is Leslee Toombs.  The husband of Stacia Toombs is Wilbert
Toombs.  ## Friends The friends of Stacia Toombs are Brian Beltran, Isiah Lutz, Leeann Hackworth, Lesley Lutz, Ryan Wang.  ##
Attributes The date of birth of Stacia Toombs is 0959-03-22.  The occupation of Stacia Toombs is actuary.  The hobby of Stacia Toombs is
finance.  The gender of Stacia Toombs is female."

---

```repl
results = search(attribute="biomedical scientist")
print(results)
```

"(1) Lannie Smock"

---

```repl
article = retrieve_article(entity="Lannie Smock")
print(article)
```

Observation: "# Lannie Smock ## Family The sons of Lannie Smock are Eli Smock, Gene Smock.  The husband of Lannie Smock is Alvaro Smock.
## Friends The friends of Lannie Smock are Williams Smock, Aida Wang, Alison Smock.  ## Attributes The date of birth of Lannie Smock
is 0867-08-24.  The occupation of Lannie Smock is biomedical scientist.  The hobby of Lannie Smock is bus spotting.  The gender of
Lannie Smock is female."

"# Eli Smock ## Family The brother of Eli Smock is Gene Smock.  The mother of Eli Smock is Lannie Smock.  The father of Eli
Smock is Alvaro Smock.  ## Friends The friends of Eli Smock are Leisa Lutz, Shelli Beltran, Vicki Hackworth, Virgil Hackworth, Alison
Smock, Brian Beltran.  ## Attributes The date of birth of Eli Smock is 0901-01-18.  The occupation of Eli Smock is retail manager.  The
hobby of Eli Smock is tether car.  The gender of Eli Smock is male."

---

```repl
gene_article = retrieve_article(entity="Gene Smock")
print(gene_article)
```

"# Gene Smock ## Family The brother of Gene Smock is Eli Smock.  The mother of Gene Smock is Lannie Smock.  The father of Gene Smock is Alvaro Smock.  The son of Gene Smock is Williams Smock.  The wife of Gene Smock is Dominique Smock.  ## Friends The
friends of Gene Smock are Leeann Hackworth, Leisa Lutz, Ricardo Hackworth, Alvaro Smock, Dominique Smock.  ## Attributes The date of
birth of Gene Smock is 0898-08-16.  The occupation of Gene Smock is immunologist.  The hobby of Gene Smock is architecture.  The gender
of Gene Smock is male."

---

```repl
article = retrieve_article(entity="Ryan Wang")
print(article)
```

"# Ryan Wang ## Family The daughter of Ryan Wang is Johnetta Wang.  The wife of Ryan Wang is Aida Wang.  ## Friends The
friends of Ryan Wang are Shelli Beltran, Stacia Toombs, Virgil Hackworth, Aida Wang.  ## Attributes The date of birth of Ryan Wang is
0982-03-17.  The occupation of Ryan Wang is chief of staff.  The hobby of Ryan Wang is fossil hunting.  The gender of Ryan Wang is
male."

"""


SYNTHWORLDS_RLM_INSTRUCTIONS = """
You are solving multi-hop questions over a corpus of documents about entities — people, places, countries, languages, organizations, etc You must reason entirely from retrieved documents to find the answer.

Your answer must be a short entity name or date (e.g. "Velthar", "28 July 1896", "Elara Windsong").

Here are some examples to understand how to use the tools available to you:

---

```repl
docs = retrieve_top_5(query="Starlight Gala founder")
print(docs[0])
```

The Starlight Gala is an annual cultural festival and charity event held in Crystalreach It was founded in 1902 by Thessan Vaylen a philanthropist and patron of the arts The Gala raises funds for the Luminara Arts Foundation and features performances in music theatre and dance Crystalreach is a large city and port The Moonlit Gala is a different event Thessan Korr is a different person The Luminara Arts Foundation is an organization

---

```repl
docs = retrieve_top_5(query="Thessan Vaylen birthplace")
print(docs[0])
```

Thessan Vaylen was born in Eldoria and later became a cultural organizer and philanthropist

---

```repl
docs = retrieve_top_5(query="Kirelle Beryndra")
print(docs[0])
print(docs[1])
```

Kirelle Beryndra died in Windhollow Windhollow is a town in Draconis County within the sovereign state of Vanthelis

Doubla has been a friend of Kirelle Beryndra and together they conquered the literary world with their works in Velthar and association with the Verdantism movement.

---

```repl
docs = retrieve_top_5(query="Draconis County named after")
print(docs)
```

'["Draconis County is a county in Vanthelis. It was established on 3 July 1862 and is named after the person Valendriel Oakshadow. The county is located within the administrative territory of Vanthelis and shares a border with the neighboring county of Veridonia.", "Vanthelis is a sovereign state and constitutional monarchy on the continent of Jorathia. Its capital is Stormhaven and the official language is Velthar.", "Veridonia is a county in Vanthelis. It was established on 15 August 1870 and is named after the person Orin Thallavar.", "Valendriel Oakshadow (born 1790, died 1855) was a military officer and statesperson who held the rank of general. He served in the Vanthelis armed forces during the Jorathian Wars.", "Oakshadow Barracks is a military installation in Stormhaven. It was named in honour of Valendriel Oakshadow and constructed in 1880."]'

---

```repl
docs = retrieve_top_5(query="Luminara River basin countries")
print(docs[0])
```

The Luminara River flows through several countries including Thundra

---

```repl
docs = retrieve_top_5(query="Thalvyn Province named after")
print(docs)
```

'["Thalvyn Province is in Thundra and is named after Caldris Veymont He spoke Velthar as his native language Thundra has Velthar as an official language Velestia is its capital Veymont Square is a park", "Thundra is a sovereign state and republic on the continent of Valtoria. The federal capital and largest city is Velestia. Velthar and Oranthor are official languages of Thundra.", "Caldris Veymont (born 2 March 1790, died 11 November 1855) was a diplomat and politician in Thundra. He spoke Kynthor and Velthar, with Velthar as his native language. He served as ambassador to Terrasylvia from 1830 to 1845.", "Velestia is a big city and the federal capital of Thundra. It is located on the Luminara River and was established in 1710.", "Veymont Square is a public park in Velestia named in honour of Caldris Veymont. It was opened in 1870."]'
"""