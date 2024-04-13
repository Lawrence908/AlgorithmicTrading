# pyStockAnalyzer
---
### Installation
How to install the program and prepare for running.
Download files
Navigate to main folder - 				“/pyStockAnalyzer”
Create a new virtual environment -			“python3 -m venv .venv”
Activate the environment - 				“source .venv/bin/activate
Install required packages - 				“pip install -r requirements.txt”


Below section only required if viewing notebooks:

	if viewing notebooks in VS Code - 

		Install Python extension into VSCode -
 		Install Jupyter extension into VSCode -

	else … follow now your IDE procure

    Install python kernel for your created venv - 	“python3 -m ipykernel install --user --name=.venv”
    Select your installed kernel to be used with your notebook - “Select Kernel” button in top right.
    Alternatively notebooks may be viewed in any application capable of handling *.ipynb files

Helpful link

Running the visual program outputs:

    Run the streamlit servers:

	“streamlit run finalPrototype.py”
	“streamlit run finalMoonShot.py”
	“streamlit run finalRecommendations.py”

    Run the notebooks with provided display(open file, select kernel, “Run All”)

 	AlgorithmicTrading.ipynb
 	moonShot2Testing.ipynb
 	posititionsTesting.ipynb
 	recommendationsTesting.ipynb

---
### Version history:
---
v 1.0

- 
- 
---
V 0.2
- shortTermSignals.py and longTermSignals.py
- short SMAs = 5, 10, 20, 50. 
- long SMAs= 50, 100, 150, 200.

Both files iterate through each ticker listed in the file specified at argv[1] (working with s+p100.txt) and do the following:
  - Proceed to download last 6 months of data from yfinance and calculate various moving avgs based on 'Adj Close'.
  - Plot these on a graph against ticker 'Adj Close' using (pyplot?)
  - Add a green arrow up, when SMAs are crossing, (change in stock momentum and overall trajectory up I believe)
  - Add a red arrow down, when SMAs are crossing, (the same but trajectory down)
  - If an arrow is marked then 'entry' column for that SMA gets a 1, or -1 to reflect.
  - Sum the 3 'entry' columns and if the last 10 days showed 'entry' at a value of 3, or -3 (all sentiment up or down) then print "Buy: " ticker, as a Buy signal, and "Sell: " ticker as a Sell signal.
  - Also save that particular chart plot to the "shortTerm" or "longTerm" folder in the "figures" folder.
  - Close the plot!
Carry on iteration


Note: Error with BRK.B ticker data download:
  - 1 Failed download:
  - ['BRK.B']: Exception('%ticker%: No timezone found, symbol may be delisted')
  - Research shows possible fix by defining dataframe timezone, but maybe markets need to be open IOT function.


Future adjustments to Trading Strategy are required:
- Refine SMA lengths,
- Refine SMA crossing signals and ensure only good signal,
- Refine condition to activate Buy or Sell signal, (more or less signals in a row).


Next V:
- Source sentiment from websites and various analysts to corroborate?
- Some calculation and technical analysis from pandas_ta to predict further sentiment ahead of time?
- Print out a nice PDF(LaTeX?) list of the data and only the important Buy/Sell/charts to pay attention to?
