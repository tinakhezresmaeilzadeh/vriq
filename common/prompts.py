demo_prompt_tir = """
Please read the following examples separated by **********. Then extract the final answer from the model's response and return it. Please only return your extracted answer content only. Please first follow these rules: 0) If the question is a multiple choice question, never extract the content of the answer, always only eturn the option letter denoting the answer. For example, if the response is F. 315, your extracted output should be F. 1) If the question is an open-ended problem, always extract the numeric value WITHOUT unit. 2) Never extract a number as a word in english, always output the numerical value, for example, if the response is "The answer is three", your extracted output should be 3. 3) If the model response is not a valid answer, please return None.4) if the model response is a list [1, 2, 3, 4, 12], please only return the content of the list 1, 2, 3, 4, 12.

**********


Question: What is the total number of items in the image?

Model response:         ```json\n{\n  \"different_patches\": [1, 2, 3, 4, 12]\n}\n```

Extracted answer: 1, 2, 3, 4, 12


**********


Question: "Please complete a maze game shown in the figure. Starting from the red ball in the top-left corner, navigate the maze to reach the green ball in the bottom-right corner. 'R' means move one step to the right, 'L' means move one step to the left, 'U' means move one step up, and 'D' means move one step down. Which of the following options can successfully lead out of the maze?\nA. RDDUDDRURLRDRUDRDDDRUUUDRRRULRRLDLDRUDRURDDDLDRDLRLLLDDURDUUDRLLDRDLDDUURULRLDRLLRDURRDURLDULLRULRUDRURLRDDRRDLU\nB. UULULDLURLUUDRUULRRUURDDLULDDUULUDUDDDDDLUUDDURLRDUDULLLRDUDRDRDR\nC. RRDDLLDDRRRRRRDDLLDDLLUULLDDDDDDRRRRUURRRRRRDDRR\nD. LURRRRDUDRRRURRDLRDRUUULURURLDLURDRLDRLDLRDLRLLRDUUUULLDDRDRURRLDR\nE. RLDDLRDRURRLDRRULLLULDUDRDULDDULDDRLRDDLRRDDRRRRRURLRDUUDLURULRUDRDDLUULRRDDLDRDUUDLDUDRRUDRLRRUDULDULURURLDUDRRDDDDURLD\nF. No answer

Model response: Thinking process here xxxxx. The correct answer is (A) RDDUDDRURLRDRUDRDDDRUUUDRRRULRRLDLDRUDRURDDDLDRDLRLLLDDURDUUDRLLDRDLDDUURULRLDRLLRDURRDURLDULLRULRUDRURLRDDRRDLU.

Extracted answer: A


**********


Question: What is the reading of the second gauge from the left in kg/cm²? Provide one decimal place (e.g., 1.1).

Model response: The reading of the second gauge from the left is 1.2 kg/cm².

Extracted answer: 1.2


**********


Question: What is the reading of the second gauge from the left in kg/cm²? Provide one decimal place (e.g., 1.1).

Model response: The reading of the second gauge from the left is 12 kg/cm².

Extracted answer: 12


**********


Question: What is the position of the black car relative to the tram?\nA. In front of the tram\nB. To the right of the tram\nC. Behind the tram\nD. To the left of the tram?

Model response: "To determine the position of the black car relative to the tram, let's analyze the image step by step:\n\n1. Identify the tram: The tram is in the center of the image, moving along the street.\n2. Look for the black car: The black car is located further back on the street, behind the tram.\n\nGiven this analysis, the black car is positioned behind the tram.\n\nThe correct answer is:\nC. Behind the tram

Extracted answer: C


**********


Question: How many cows are there in total in the picture?\n

Model response: There are give cows in total in the picture.

Extracted answer: 5


**********


Question: How many dogs are there in total in the picture?\n

Model response: Six dogs

Extracted answer: 6


**********


Question: How many degrees should you rotate this image CLOCKWISE to restore it to its original orientation?\n\nA. 25°\nB. 55°\nC. 85°\nD. 70°\nE. 35°\nF. 40°\n

Model response: To determine how many degrees the image should be rotated clockwise to restore it to its original orientation, we need to analyze the angle at which the image is currently tilted.\n\n1. **Identify the Tilt Angle**: The image appears to be tilted at an angle. By visually estimating the angle, it looks like the image is tilted approximately 35 degrees counterclockwise from the horizontal.\n\n2. **Determine the Rotation Needed**: To restore the image to its original orientation, we need to rotate it clockwise by the same angle it is currently tilted counterclockwise. Therefore, we need to rotate the image 35 degrees clockwise.\n\n3. **Select the Correct Option**: From the given options, the correct choice is:\n\nE. 35°\n\nSo, the image should be rotated 35 degrees clockwise to restore it to its original orientation.

Extracted answer: E


**********


Question: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: .......the correct answer is:\n\n\\boxed{B}

Extracted answer: B


**********


Question: The difference between the lowest blue bar and the lowest gray bar is at ( ).\nA. [300, 350]\nB. [350, 400]\nC. [400, 450]\nD. [450, 500]\nE. [500, 550]\nF. [550, 600]

Model response: To determine the difference between the lowest blue bar and the lowest gray bar, we need to estimate the values of these bars from the chart.\n\n1. **Identify the lowest blue bar:**\n   - The lowest blue bar appears to be around 250.\n\n2. **Identify the lowest gray bar:**\n   - The lowest gray bar appears to be around 550.\n\n3. **Calculate the difference:**\n   - Difference = Value of the lowest gray bar - Value of the lowest blue bar\n   - Difference = 550 - 250 = 300\n\n4. **Determine the range:**\n   - The difference of 300 falls within the range [300, 350].\n\nTherefore, the correct answer is:\nA. [300, 350]

Extracted answer: A


**********


Question: When Pretrain Loss=1.75, SFT minus SFT 1/2 is about ()\nA. 2\nB. 5\nC. 7\nD. 10\nE. 14\nF. 17\nG. -1

Model response:  "To determine the difference between SFT and SFT 1/2 when Pretrain Loss = 1.75, we need to look at the corresponding values on the graph for these two methods at that specific pretrain loss.\n\n1. Locate the point on the x-axis where Pretrain Loss = 1.75.\n2. Find the y-axis values for SFT and SFT 1/2 at this point.\n\nFrom the graph:\n- SFT (solid blue line) is approximately at 50.\n- SFT 1/2 (dashed blue line) is approximately at 48.\n\nNow, calculate the difference:\n\\[ \\text{Difference} = \\text{SFT} - \\text{SFT 1/2} = 50 - 48 = 2 \\]\n\nTherefore, the correct answer is:\nA. 2"

Extracted answer: A


***********


Question: Please complete a maze game shown in the figure. Starting from the red ball in the top-left corner, navigate the maze to reach the green ball in the bottom-right corner. 'R' means move one step to the right, 'L' means move one step to the left, 'U' means move one step up, and 'D' means move one step down. Which of the following options can successfully lead out of the maze?\nA. DURDLDDDRUUDDDDLDDLRUDDRDDRLURDURDLLULUDUUUDRLRUDLRRDUUUDLDLLUDLUULDDLURLDUDLUDLDLRULRRULDRRRUURUDULRUDRRDDLRURRUURRDURDRULL\nB. RRDRDRLRUULLDRUUDUDLDRDRLDRUUUDUDUURDDRUDLRRLDLRRLDUURLDLDDDUDDDLRDRUUDURDUURUDURRLRLDU\nC. DDRRRRDD\nD. RLLLRLDUDDDDRUUDLLRURDDDLRU\nE. LDRULDULLUDDLLURUDLLRRRRDDLDRURRRRRDRLRDLRUUDULUUDLUUUDRLDDDDUDUDDLLULURRULRDLRURDDUUDRRULDDDLDDURULUDRLRR\nF. No answer

Model response: To solve this maze game, we need to navigate from the red ball at the top-left corner to the green ball at the bottom-right corner using the given instructions. Let's analyze each option step by step.\n\n### Option A: DURDLDDDRUUDDDDLDDLRUDDRDDRLURDURDLLULUDUUUDRLRUDLRRDUUUDLDLLUDLUULDDLURLDUDLUDLDLRULRRULDRRRUURUDULRUDRRDDLRURRUURRDURDRULL\n- Start at (1,1)\n- Move down: (1,2)\n- Move right: (2,2)

Extracted answer: None
"""


demo_prompt_mathvista = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: Two.

Extracted answer: 2
"""

demo_prompt_superclevr_counting = """
Question: How many different items are there in the image?

Model response: 6

Extracted answer: 6

Question: How many different items are there in the image?

Model response: This is an image of xxx, there are 10 differnt items in the image.

Extracted answer: 10

"""