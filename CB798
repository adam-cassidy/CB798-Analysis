import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns;
sns.set(style="ticks", color_codes=True)
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

# for ML and advanced analysis
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

filename = r"D:\_Python\Py\Uni\hotel_bookings.csv"
df = pd.read_csv(filename)


# **** Data Cleaning ****

# Convert arrival month to numeric
monthConvert = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9,
                'October': 10, 'November': 11, 'December': 12}
df.arrival_date_month = df.arrival_date_month.map(monthConvert)

# Concatenate day/month/year into useable format
df['arrival_date'] = df['arrival_date_year'].astype(str) + df['arrival_date_month'].astype(str).str.zfill(2) + \
                     df['arrival_date_day_of_month'].astype(str).str.zfill(2)
df['arrival_date'] = pd.to_datetime(df['arrival_date'], format='%Y%m%d')

# Remove the unnecessary time stamp from date formatting
df['arrival_date'] = df['arrival_date'].dt.date

# Separates cancelled holidays for analysis
df['is_not_canceled'] = df['is_canceled']
df['is_not_canceled'].replace({0: 1, 1: 0}, inplace=True)

# **** Basic Overview ****
# To begin, we will look at some basic figures.

# Number of holidays cancelled
tempCancellations = ['Not Cancelled', 'Cancelled']
dfCanc = df
#dfCanc.replace({0: 'Not Cancelled', 1: 'Cancelled'})  # , inplace=True)
x1 = dfCanc['is_canceled'].value_counts()

x1.sort_index(ascending=False, inplace=True)
cancellednotcancelled = ['Cancelled', 'Not Cancelled']
ax = sns.barplot(x=x1.index, y=x1, order=[1, 0])
ax.set_ylabel("Number of Bookings")
ax.set_xlabel("Group")
plt.xticks(np.arange(2), cancellednotcancelled)
ax.set_title("Raw Cancelled Holidays")
plt.show()

# Holidays cancelled time series - with rolling mean
df.sort_values(by=['arrival_date'], na_position='first', inplace=True)
timeSeriesCancelled = df.groupby("arrival_date")["is_canceled"].sum()
timeSeriesNotCancelled = df.groupby("arrival_date")["is_not_canceled"].sum()

rolling_mean = timeSeriesCancelled.rolling(window=7).mean()
rolling_mean2 = timeSeriesNotCancelled.rolling(window=7).mean()
fig, ax1 = plt.subplots()

ax1.plot(df['arrival_date'].unique(), rolling_mean, label="Cancellations (Daily Avg)", color="orange")
ax1.plot(df['arrival_date'].unique(), rolling_mean2, label="Successful Bookings (Daily Avg)", color="Magenta")
ax1.legend(loc="upper right")
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=30))
ax1.set_ylabel("Number of Bookings", fontsize=14)
ax1.set_xlabel("Date", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=12)
ax1.set_title("Time Series of Cancellations and Bookings (7 Day Rolling Avg)", fontsize=16)
fig.set_size_inches(18, 6)
plt.show()
#TODO: RECODE 0 and 1 X AXIS

# Are there any notable spikes or trends - either days of the week or months?
# cancAvg = df['is_canceled'].sum() / len(df) === 62.9% bookings cancelled
# x = rolling_mean.mean() suggests average of 94 cancellations per booking day.
# Setting a minimum of 94, we can plot and examine dates only with 90+ cancellations

# Add daily avg to main dataframe
df['avg_cancelations_day'] = df.groupby("arrival_date")["is_canceled"].transform('sum')

# Removes any days with less than average cancellations
df94 = df[df['avg_cancelations_day'] > 93]

# Replot same graph with new criteria
timeSeriesCancelled94 = df94.groupby("arrival_date")["is_canceled"].sum()
timeSeriesNotCancelled94 = df94.groupby("arrival_date")["is_not_canceled"].sum()

rolling_mean = timeSeriesCancelled94.rolling(window=7).mean()
rolling_mean2 = timeSeriesNotCancelled94.rolling(window=7).mean()
fig, ax2 = plt.subplots()

ax2.plot(df94['arrival_date'].unique(), rolling_mean, label="Cancellations (Daily Avg)", color="orange")
ax2.plot(df94['arrival_date'].unique(), rolling_mean2, label="Successful Bookings (Daily Avg)", color="Magenta")
ax2.legend(loc="upper right")
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=30))
ax2.set_ylabel("Number of Bookings", fontsize=14)
ax2.set_xlabel("Date", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=12)
ax2.set_title("Time Series of Cancellations and Bookings (Above Avg Only) (7 Day Rolling Avg)", fontsize=16)
fig.set_size_inches(18, 6)
plt.show()

# Or without a rolling average...
rolling_mean = timeSeriesCancelled94.rolling(window=1).mean()
rolling_mean2 = timeSeriesNotCancelled94.rolling(window=1).mean()

fig, ax3 = plt.subplots()
ax3.plot(df94['arrival_date'].unique(), rolling_mean, label="Cancellations (Daily Avg)", color="orange")
ax3.plot(df94['arrival_date'].unique(), rolling_mean2, label="Successful Bookings (Daily Avg)", color="Magenta")
ax3.legend(loc="upper right")
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=30))
ax3.set_ylabel("Number of Cancellations", fontsize=14)
ax3.set_xlabel("Date", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=12)
ax3.set_title("Time Series of Cancellations and Bookings (Above Avg Only, No Rolling)", fontsize=16)
fig.set_size_inches(18, 6)
plt.show()

# We can clearly see that there are frequent spikes, however these spikes are typically
# In line with booking frequency.

# We can break it down by month and day of the week.

# Month
# Recode month numbers back to string for graphing
monthConvertBack = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
                    9: 'September',
                    10: 'October', 11: 'November', 12: 'December'}
df.arrival_date_month = df.arrival_date_month.map(monthConvertBack)

# Set month to datetime so it can be ordered by month and not alphabetically
# Set ordered coding for months
df['arrival_date_month_ordered'] = df['arrival_date_month']
df['arrival_date_month_ordered'] = df.arrival_date_month_ordered.map(monthConvert)
df['arrival_date_month_ordered'] = df['arrival_date_month_ordered'].astype(int)

cancellationMonth = df.groupby("arrival_date_month")["is_canceled"].sum()
cancellationNotMonth = df.groupby("arrival_date_month")["is_not_canceled"].sum()

# Graph it
labels = cancellationMonth.index
x2 = np.arange(len(labels))
width = 0.4
fig, ax4 = plt.subplots()

monthOrder = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
              'October', 'November', 'December']
cancellationMonth.index = pd.CategoricalIndex(cancellationMonth.index, categories=monthOrder, ordered=True)
cancellationNotMonth.index = pd.CategoricalIndex(cancellationNotMonth.index, categories=monthOrder, ordered=True)
cancellationMonth = cancellationMonth.sort_index()
cancellationNotMonth = cancellationNotMonth.sort_index()

ax4.bar(x2 - width / 2, cancellationMonth, width, label='Cancellations', align='center', tick_label=monthOrder)
ax4.bar(x2 + width / 2, cancellationNotMonth, width, label='Successful', align='center', tick_label=monthOrder)
ax4.set_ylabel("Frequency")
ax4.set_xlabel("Day of Month")
ax4.set_title("Frequency of Bookings and Cancellations by Month of Year")
plt.xticks(rotation=45, ha="right", fontsize=12)
ax4.legend()
plt.show()

# Stay Length
df["total_stay"] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
lengthStay = df['total_stay'].value_counts()
lengthStay.sort_index(ascending=True, inplace=True)
lengthStay.drop(lengthStay.head(1).index, inplace=True)
lengthStay = lengthStay.head(14)
cancellationStay = df.groupby("total_stay")["is_canceled"].sum()
cancellationNotStay = df.groupby("total_stay")["is_not_canceled"].sum()
# Removes dates with 0 night stay
cancellationStay.drop(cancellationStay.head(1).index, inplace=True)
cancellationNotStay.drop(cancellationNotStay.head(1).index, inplace=True)
# Removes dates with 0 night stay

cancellationStay = cancellationStay.head(14)
cancellationNotStay = cancellationNotStay.head(14)
#Displays only stays up to 2 weeks in length; further provides too small a sample size

#cancellationStay = (cancellationStay / lengthStay) * 100
#cancellationNotStay = (cancellationNotStay / lengthStay) * 100

# Graph it
labels = cancellationStay.index
x2 = np.arange(len(labels))
width = 0.4
fig, ax5 = plt.subplots()

lengthOrder = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13","14"]
#ax5.bar(x2 - width / 2, cancellationStay, width, label='Cancellations', align='center')  # , tick_label=lengthOrder)
#ax5.bar(x2 + width / 2, cancellationNotStay, width, label='Successful', align='center')  # , tick_label=lengthOrder)
ax5.plot(cancellationStay, label='Cancellations')
ax5.plot(cancellationNotStay, label='Successful')
#ax5.plot(cancellationStay)
ax5.set_ylabel("Frequency")
#ax5.set_ylabel("% Value")
ax5.set_xlabel("Length of Stay")
ax5.set_title("Frequency of Bookings and Cancellations by Length of Stay")
plt.xticks(np.arange(15), lengthOrder)
ax5.legend()

plt.show()


# Deposits
deposit = df['deposit_type'].value_counts()
# Merely 15,000 of over 100,000 inputs have a deposit, so raw values are useless to us here

# No deposit / non refund / refundable will be turned into a % of their values
cancellationDeposit = df.groupby("deposit_type")["is_canceled"].sum()
cancellationNotDeposit = df.groupby("deposit_type")["is_not_canceled"].sum()

cancellationDeposit = (cancellationDeposit / deposit) * 100
cancellationNotDeposit = (cancellationNotDeposit / deposit) * 100
# Turns raw values into %

# Graph it
labels = cancellationDeposit.index
x2 = np.arange(len(labels))
width = 0.4
fig, ax6 = plt.subplots()

depositOrder = ["No Deposit", "Non Refund", "Refundable"]
ax6.bar(x2 - width / 2, cancellationDeposit, width, label='Cancellations', align='center')  # , tick_label=lengthOrder)
ax6.bar(x2 + width / 2, cancellationNotDeposit, width, label='Successful', align='center')  # , tick_label=lengthOrder)
ax6.set_ylabel("Frequency (%)")
ax6.set_xlabel("Deposit Type")
ax6.set_title("Frequency of Bookings and Cancellations by Deposit Type")
plt.xticks(np.arange(3), depositOrder)
ax6.legend()

plt.show()

# Previous cancellations
numCancellations = df['previous_cancellations'].value_counts()
# Over 95% of respondents had never cancelled before. To avoid anomalous data,
# this will not be considered

# Repeat guest
NumRepeat = df['is_repeated_guest'].value_counts()
# Overwhelming majority of respondents are not repeat guests, but enough are to make
# calculation viable. Values as % for same reason as deposits.

cancellationRepeat = df.groupby("is_repeated_guest")["is_canceled"].sum()
cancellationNotRepeat = df.groupby("is_repeated_guest")["is_not_canceled"].sum()

cancellationRepeat = (cancellationRepeat / NumRepeat) * 100
cancellationNotRepeat = (cancellationNotRepeat / NumRepeat) * 100
# Turns raw values into %

# Graph it
labels = cancellationRepeat.index
x2 = np.arange(len(labels))
width = 0.4
fig, ax7 = plt.subplots()

RepeatOrder = ["Not Repeat Guest", "Repeat Guest"]
ax7.bar(x2 - width / 2, cancellationRepeat, width, label='Cancellations', align='center')  # , tick_label=lengthOrder)
ax7.bar(x2 + width / 2, cancellationNotRepeat, width, label='Successful', align='center')  # , tick_label=lengthOrder)
ax7.set_ylabel("Frequency (%)")
ax7.set_xlabel("Repeat Customer Type")
ax7.set_title("Frequency of Bookings and Cancellations by Repeat Customers")
plt.xticks(np.arange(2), RepeatOrder)
ax7.legend()
#TODO: Make sure these values are actually correct.
plt.show()

# Guests
# To start, should we even consider children/babies? Is there enough data?
dfTEST = df['children'].isna()
df['children'] = df['children'].fillna(0) # remove NaN values
df['children'] = df['children'].astype(int) # children coded as float in original dataset

children = df['children'].value_counts()
babies = df['babies'].value_counts()
adults = df['adults'].value_counts()
df['total_guests'] = df['children'] + df['babies'] + df['adults']
df = df[df.total_guests != 0]
totalGuests = df['total_guests'].value_counts()

# Remove obscure and potentially anomolous data, only consider results with 100+ values
children = children.head(4)
babies = babies.head(3)
adults = adults.head(4)
adults.sort_index(inplace=True)
totalGuests = totalGuests.head(5)

# Create percentages for each of the four categories
cancellationBabies = df.groupby("babies")["is_canceled"].sum()
cancellationNotBabies = df.groupby("babies")["is_not_canceled"].sum()
cancellationBabies = (cancellationBabies / babies) * 100
cancellationNotBabies = (cancellationNotBabies / babies) * 100
cancellationBabies = cancellationBabies.head(3)
cancellationNotBabies = cancellationNotBabies.head(3)

cancellationChildren = df.groupby("children")["is_canceled"].sum()
cancellationNotChildren = df.groupby("children")["is_not_canceled"].sum()
cancellationChildren = (cancellationChildren / children) * 100
cancellationNotChildren = (cancellationNotChildren / children) * 100
cancellationChildren = cancellationChildren.head(4)
cancellationNotChildren = cancellationNotChildren.head(4)

cancellationAdults = df.groupby("adults")["is_canceled"].sum()
cancellationNotAdults = df.groupby("adults")["is_not_canceled"].sum()
cancellationAdults = (cancellationAdults / adults) * 100
cancellationNotAdults = (cancellationNotAdults / adults) * 100
cancellationAdults = cancellationAdults.head(4)
cancellationNotAdults = cancellationNotAdults.head(4)

cancellationTotal = df.groupby("total_guests")["is_canceled"].sum()
cancellationNotTotal = df.groupby("total_guests")["is_not_canceled"].sum()
cancellationTotal = (cancellationTotal / totalGuests) * 100
cancellationNotTotal = (cancellationNotTotal / totalGuests) * 100
cancellationTotal = cancellationTotal.head(5)
cancellationNotTotal = cancellationNotTotal.head(5)

# 4 subplot figure to compare guest types
fig, ((ax9, ax10), (ax11, ax12)) = plt.subplots(2, 2)
fig.suptitle("Frequency of Bookings and Cancellations by Guests")
GuestOrder = ["0","1","2","3"]

cancellationBabies.plot.bar(grid=False, ax=ax9, width=width, position=0, label='_nolegend_')
cancellationNotBabies.plot.bar(grid=False, ax=ax9, width=width, position=1, color='#ff7f0e', label='_nolegend_')
cancellationChildren.plot.bar(grid=False, ax=ax10, width=width, position=0, label='_nolegend_')
cancellationNotChildren.plot.bar(grid=False, ax=ax10, width=width, position=1, color='#ff7f0e', label='_nolegend_')
cancellationAdults.plot.bar(grid=False, ax=ax11, width=width, position=0, label='_nolegend_')
cancellationNotAdults.plot.bar(grid=False, ax=ax11, width=width, position=1, color='#ff7f0e', label='_nolegend_')
cancellationTotal.plot.bar(grid=False, ax=ax12, width=width, position=0, label='Cancellations')
cancellationNotTotal.plot.bar(grid=False, ax=ax12, width=width, position=1, color='#ff7f0e', label='Successful')
ax9.set_ylabel("Frequency (%)")
ax11.set_ylabel("Frequency (%)")
ax9.set_xlabel("Babies")
ax10.set_xlabel("Children")
ax11.set_xlabel("Adults")
ax12.set_xlabel("Total")
ax9.tick_params(labelrotation=0)
ax10.tick_params(labelrotation=0)
ax11.tick_params(labelrotation=0)
ax12.tick_params(labelrotation=0)

ax9.set_ylim([0, 100])
ax10.set_ylim([0, 100])
ax11.set_ylim([0, 100])
ax12.set_ylim([0, 100])

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='upper left', borderaxespad=0)
fig.set_size_inches(8, 6)
fig.tight_layout()

plt.show()

# HOME Country OF GUESTS
# THIS IS NOT LOCATION OF HOTEL!!! THIS IS NATIONALITY!!!!

# Remove data where country is not specified
numCountry = df['country'].value_counts()
# 177 different countries listed. Only considering countries highest 20 countries to avoid anomalous data
numCountry = numCountry.head(20)
a = numCountry.index

cancellationCountry = df.groupby("country")["is_canceled"].sum()
cancellationNotCountry = df.groupby("country")["is_not_canceled"].sum()

cancellationCountry = cancellationCountry[cancellationCountry.index.isin(a)]
cancellationNotCountry = cancellationNotCountry[cancellationNotCountry.index.isin(a)]
# Removes all values that are not in numCountry (not highest 20)

# sort numCountry so all three series follow the same order. Otherwise, following calculations are completely off.
numCountry.sort_index(inplace=True)

cancellationCountry = (cancellationCountry / numCountry) * 100
cancellationNotCountry = (cancellationNotCountry / numCountry) * 100
# Turns raw values into %

# Graph it
labels = cancellationCountry.index
x2 = np.arange(len(labels))
width = 0.45
fig, ax13 = plt.subplots()

# Order graph highest to lowest
# To avoid data getting mixed, must form a dataframe in this case
countryOrderBasic = ["AUT", "BEL", "BRA", "CHE", "CHN", "CN", "DEU", "ESP", "FRA", "GBR", "IRL", "ISR", "ITA", "NLD", "NOR",
               "POL", "PRT", "RUS", "SWE", "USA"]
countryOrder = pd.Series(countryOrderBasic)
frame = {"Cancelled": cancellationCountry, "Not_Cancelled": cancellationNotCountry}
dfCountry = pd.DataFrame(frame)
dfCountry.sort_values(by=['Not_Cancelled'], inplace=True)

ax13.bar(x2 - width / 2, dfCountry.Cancelled, width, label='Cancellations', align='center')
ax13.bar(x2 + width / 2, dfCountry.Not_Cancelled, width, label='Successful', align='center')
ax13.set_ylabel("Frequency (%)")
ax13.set_xlabel("Country")
ax13.set_title("Frequency of Bookings and Cancellations by Country")
plt.xticks(np.arange(20), dfCountry.index)
ax13.tick_params(labelrotation=45)
ax13.legend()
plt.show()

# Lead time

numLead = df['lead_time'].value_counts()

cancellationLeadFreq = df.groupby("lead_time")["is_canceled"].sum()
cancellationNotLeadFreq = df.groupby("lead_time")["is_not_canceled"].sum()
# Separate series for frequency so both freq and % can be plotted

cancellationLead = (cancellationLeadFreq / numLead) * 100
cancellationNotLead = (cancellationNotLeadFreq / numLead) * 100
# Turns raw values into %

# Graph it
labels = cancellationLead.index
x2 = np.arange(len(labels))
width = 0.45
fig, axes = plt.subplots(2, figsize=(10,6), sharex=True)
ax140 = axes[0]
divider = make_axes_locatable(ax140)
ax142 = divider.new_vertical(size="100%", pad=0.04)
fig.add_axes(ax142)

ax140.scatter(x= numLead.index, y= numLead.values, color="magenta", label="Cancellations (Frequency)")
m, b = np.polyfit(numLead.index, numLead.values, 1) # line of best fit
ax140.plot(numLead.index, m * numLead.index + b)
ax140.set_xlabel("Lead Time (Days) (Frequency)")
#ax14[1].semilogy(2)
ax140.set_ylim(0, 1000)
ax140.spines['top'].set_visible(False)
ax142.scatter(x= numLead.index, y= numLead.values, color="magenta", label="Cancellations (Frequency)")
ax142.set_ylim(1000, 6300)
ax142.tick_params(bottom=False, labelbottom=False)
ax142.spines['bottom'].set_visible(False)
#ax14[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax140.set_ylabel("No. Bookings")



d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax142.transAxes, color='k', clip_on=False)
ax142.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax142.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax140.transAxes)  # switch to the bottom axes
ax140.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax140.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

axes[1].scatter(x = cancellationLead.index, y= cancellationLead.values, color="orange", label="Cancellations (%)")
m, b = np.polyfit(cancellationLead.index, cancellationLead.values, 1) # line of best fit
axes[1].plot(cancellationLead.index, m * cancellationLead.index + b)
axes[1].set_xlabel("Lead Time (Days) - Cancellations (%)")
axes[1].set_ylabel("% Value")

fig.suptitle("Cancellations by Lead Time & Lead Time Frequency")

plt.show()

"""labels = cancellationLead.index
x2 = np.arange(len(labels))
width = 0.45
fig, ax14 = plt.subplots(2, figsize=(10,6), sharex=True)

ax14[0].scatter(x = cancellationLead.index, y= cancellationLead.values, color="orange", label="Cancellations (%)")
m, b = np.polyfit(cancellationLead.index, cancellationLead.values, 1) # line of best fit
ax14[0].plot(cancellationLead.index, m * cancellationLead.index + b)
ax14[0].set_xlabel("Lead Time (Days) - Cancellations (%)")
ax14[0].set_ylabel("% Value")

ax14[1].scatter(x= numLead.index, y= numLead.values, color="magenta", label="Cancellations (Frequency)")
m, b = np.polyfit(numLead.index, numLead.values, 1) # line of best fit
ax14[1].plot(numLead.index, m * numLead.index + b)
ax14[1].set_xlabel("Lead Time (Days) (Frequency)")
#ax14[1].semilogy(2)
#ax14[1].set_ylim(0, 2000)
#ax14[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax14[1].set_ylabel("Frequency")

fig.suptitle("Cancellations by Lead Time & Lead Time Frequency")

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='upper left', borderaxespad=0)

plt.show()"""

# Booking room category

# Create a boolean column to clarify whether customer received booked room type
df['room_choice'] = df['assigned_room_type'] # placeholder column for shape
df['room_choice'] = np.where(df['assigned_room_type'] == df['reserved_room_type'], 1, 0)
# Recode boolean
"""df['room_choice'] = df['room_choice'].replace("TRUE", 1)
df['room_choice'] = df['room_choice'].replace("FALSE", 0)"""

numRoom = df['room_choice'].value_counts()

cancellationRoom = df.groupby("room_choice")["is_canceled"].sum()
cancellationNotRoom = df.groupby("room_choice")["is_not_canceled"].sum()

cancellationRoom = (cancellationRoom / numRoom) * 100
cancellationNotRoom = (cancellationNotRoom / numRoom) * 100

# Graph
labels = cancellationRoom.index
x2 = np.arange(len(labels))
width = 0.45
fig, ax15 = plt.subplots()

roomOrder = ["Correct Room", "Incorrect Room"]

ax15.bar(x2 - width / 2, cancellationRoom, width, label='Cancellations', align='center')
ax15.bar(x2 + width / 2, cancellationNotRoom, width, label='Successful', align='center')
ax15.set_ylabel("Frequency (%)")
ax15.set_xlabel("Was Customer Assigned Requested Room?")
ax15.set_title("Frequency of Bookings and Cancellations by Assigned Room Choice")
plt.xticks(np.arange(2), roomOrder)
ax15.legend()
plt.show()

# ADR
# Firstly need to remove ADR values of 0
dfADR = df
dfADR = dfADR[(dfADR.adr > 1)]
dfADR = dfADR[(dfADR.adr < 550)]
adrOrder = [">50", "51-75", "76-100", "101-125", "126-150", "151-175", "176-200", "<200"]

# Reassign values into a range
dfADR['adr'] = np.where(dfADR['adr'] <= 50, 1, dfADR['adr'])
dfADR['adr'] = np.where(dfADR['adr'].between(50.01,75), 2, dfADR['adr'])
dfADR['adr'] = np.where(dfADR['adr'].between(75.01,100), 3, dfADR['adr'])
dfADR['adr'] = np.where(dfADR['adr'].between(100.01,125), 4, dfADR['adr'])
dfADR['adr'] = np.where(dfADR['adr'].between(125.01,150), 5, dfADR['adr'])
dfADR['adr'] = np.where(dfADR['adr'].between(150.01,175), 6, dfADR['adr'])
dfADR['adr'] = np.where(dfADR['adr'].between(175.01,200), 7, dfADR['adr'])
dfADR['adr'] = np.where(dfADR['adr'] > 200 ,8, dfADR['adr'])

# Do the usual setup of series
numADR = dfADR['adr'].value_counts()

cancellationADR = dfADR.groupby("adr")["is_canceled"].sum()
cancellationNotADR = dfADR.groupby("adr")["is_not_canceled"].sum()

# Graph
ax21 = sns.barplot(x=numADR.index, y=numADR.values)
ax21.set_ylabel("Frequency of ADR")
ax21.set_xlabel("ADR Category")
plt.xticks(np.arange(8), adrOrder)
ax21.set_title("ADR Rates")
plt.show()

# % Conversion
cancellationADR = (cancellationADR / numADR) * 100
cancellationNotADR = (cancellationNotADR / numADR) * 100
adrOrder = ["0", "<50", "51-75", "76-100", "101-125", "126-150", "151-175", "176-200", ">200"]

# Need to now categorise the data due to such variances with our percentages
# First, convert ADR column to int
dfADR['adr'] = dfADR['adr'].astype(int)



labels = cancellationADR.index
x2 = np.arange(len(labels))
width = 0.4
fig, ax20 = plt.subplots()


#ax5.bar(x2 - width / 2, cancellationStay, width, label='Cancellations', align='center')  # , tick_label=lengthOrder)
#ax5.bar(x2 + width / 2, cancellationNotStay, width, label='Successful', align='center')  # , tick_label=lengthOrder)
#ax5.plot(cancellationStay, label='Cancellations')
#ax5.plot(cancellationNotStay, label='Successful')
ax20.plot(cancellationADR)
#ax5.set_ylabel("Frequency")
ax20.set_ylabel("% Value")
ax20.set_xlabel("ADR")
ax20.set_title("Percentage of Cancellations by ADR")
plt.xticks(np.arange(9), adrOrder)
ax20.legend()

plt.show()

# **** Advanced analytics ****

# https://www.kaggle.com/vssseel/eda-various-ml-models-and-nn-with-roc-curves
# According to Kaggle user's analysis (l.84), Random Forest provides the highest accuracy score for this data set

# structure for decision tree based off of https://www.kaggle.com/abdulwaheedsoudagar/eda-modelling-rf-xbg-and-keras-nn

# Before using Kaggle user's Random Forest, we must recode data for it to work
# All strings must be recoded to float/int OR dropped.
df['hotel'] = df['hotel'].replace("Resort Hotel", 1)
df['hotel'] = df['hotel'].replace("City Hotel", 2)
df.arrival_date_month = df.arrival_date_month.map(monthConvert)

# Remaining columns (of lesser importance) can be recoded automatically with LabelEncoder
le = LabelEncoder()
df['meal'] = le.fit_transform(df['meal'])
df['country'] = le.fit_transform(df['country'])
df['deposit_type'] = le.fit_transform(df['deposit_type'])
df['reserved_room_type'] = le.fit_transform(df['reserved_room_type'])
df['assigned_room_type'] = le.fit_transform(df['assigned_room_type'])

# Columns irrelevant to us based on EDA can just be dropped completely from here
df.drop(['reservation_status'], inplace=True, axis=1) # Not irrelevant per se, but replica of cancellation column
df.drop(['market_segment'], inplace=True, axis=1)
df.drop(['distribution_channel'], inplace=True, axis=1)
df.drop(['customer_type'], inplace=True, axis=1)
df.drop(['reservation_status_date'], inplace=True, axis=1)
df.drop(['arrival_date'], inplace=True, axis=1)
df.drop(['agent'], inplace=True, axis=1)
df.drop(['company'], inplace=True, axis=1)
df.drop(['is_not_canceled'], inplace=True, axis=1)

# Build a forest and compute the impurity-based feature importances
def training(model,X_train, y_train):
    return model.fit(X_train, y_train)


X = df.drop(["is_canceled"], axis=1)
y = df["is_canceled"]

dfColumns = df.drop(["is_canceled"], axis=1).head(1)
indices = ['Hotel', 'Lead Time', 'Arrival Date (Year)', 'Arrival Date (Month)', 'Arrival Date Week No.',
             'Arrival Date (Day of Month)', 'Stay (Weekend Nights)', 'Stay (Week Nights)',
             'Adults', 'Children', 'Babies', 'Meal', 'Country', 'Repeat Guest', 'Previous Cancellations',
             'Previously Not Cancelled', 'Reserved Room Type', 'Assigned Room Type', 'Booking Changes',
             'Deposit Type', 'Days in Wait List', 'ADR', 'Car Parking Spaces', 'Total Special Requests', 'Avg Cancellation Day',
             'Avg Date Month Ordered', 'Total Stay', 'Total Guests', 'Room Choice']

# https://machinelearningmastery.com/calculate-feature-importance-with-python/
model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
importance = importance * 100

# summarise feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
ax16 = sns.barplot(x=importance, y=indices)
for bar in ax16.patches:
    if bar.get_width() > 5:
        bar.set_color('blue')
    else:
        bar.set_color('grey')
# Sets colour coding on plot to highlight higest/lowest values

ax16.set_ylabel("Variable")
ax16.set_xlabel("Feature Importance (%)")
ax16.set_title("Feature Importance Random Forest")
plt.show()

# Focus on advanced analytics for lead time, country, deposit type

model = LogisticRegression()
model.fit(X, y)
importance = model.coef_[0]
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
ax17 = sns.barplot(x=importance, y=indices)

for bar in ax17.patches:
    if bar.get_width() > 0.5:
        bar.set_color('blue')
    elif bar.get_width() < -0.1:
        bar.set_color('red')
    else:
        bar.set_color('grey')
# Sets colour coding on plot to highlight higest/lowest values

ax17.set_ylabel("Correlation")
ax17.set_xlabel("Variables")
ax17.set_title("Feature Importance Logistic Regression")
ax17.set_xlim([-0.25, 0.5])
plt.show()

"""plt.bar([x for x in range(len(importance))], importance)
plt.xticks(np.arange(len(importance)), indices, rotation=90)
plt.suptitle("Logistic Regression Feature Importance")
plt.show()"""

correlation = df.corr()["is_canceled"]
correlation.drop(['is_canceled'], inplace=True)
# Remove the correlations it does with itself (that provide perfect 1, -1 values)

correlation.index = indices


ax18 = sns.barplot(x=correlation.values, y=correlation.index)
# Colour code highest and lowest values
for bar in ax18.patches:
    if bar.get_width() > 0.2:
        bar.set_color('blue')
    elif bar.get_width() < -0.1:
        bar.set_color('red')
    else:
        bar.set_color('grey')
# Sets colour coding on plot to highlight higest/lowest values

ax18.set_ylabel("Correlation")
ax18.set_xlabel("Variables")
ax18.set_title("Correlation of all Dataset Variables")
ax18.set_xlim([-0.25, 0.5])
plt.show()
