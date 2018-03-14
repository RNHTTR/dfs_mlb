# A Machine Learning Approach to Daily Fantasy Baseball

Baseball (and thus fantasy baseball) generates large amounts of data on its players. This data can be harnessed to generate predictions about a given player's performance, which can in turn be used as an advantage in selecting a fantasy baseball team on DraftKings, FanDuel, or other fantasy sports site.

Predicting a player's value is not enough, though. Each player is given a dollar value associated with them. The total value of a team (the sum of the value of each players on a team) cannot be exceeded, so one most strive to get the most valuable team while not exceeding this artificial "salary cap". Additionally, one must player a certain number of players at a particular position (e.g. three outfielders, two pitchers, and so on). This requires a modified implementation of the knapsack algorithm to select the best team satisfying the salary cap and positional parameters.

So, this application strives to predict players' fantasy points as well as select the best (or near-best) team given a salary cap and positional requirements.

## Implementation

### Get the Data

Shout out to rotoguru1.com. Run GetData.py to store the player data in a csv file.

```
cd data
python GetData.py
```

### Format the Data

Edit data/FormatData.py to get a subset of the raw data that you intend to pass through the Neural Network. Then run FormatData.py to generate a csv file with formatted data.

```
cd data
python FormatData.py
```

### Data Playground

Not sure exactly how this will go yet!


### Run the Neural Network

From the root directory, run DenseNeuralNetwork.py

```
python DenseNeuralNetwork.py
```

<!-- ## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).  -->

## Authors

* **Ryan Hatter** - *Not for Reuse*

<!-- See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project. -->

<!-- ## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details -->

## Acknowledgments

* Hat tip to rotoguru1.com for the great data.
