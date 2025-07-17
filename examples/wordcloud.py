from agent_toolkit.visualize import make_wordcloud

problem = """
Veerman Furniture Company makes three kinds of office furniture: chairs, desks, and tables.
Each product requires some labor in the parts fabrication department, the assembly department, and the shipping department.
The furniture is sold through a regional distributor, who has estimated the maximum potential sales for each product.
Finally, the accounting department has provided some data showing the profit contributions on each product.
"""

make_wordcloud(problem)
# This code generates a word cloud from the provided problem text.