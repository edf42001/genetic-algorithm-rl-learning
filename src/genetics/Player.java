package genetics;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.UnitTemplate;
import edu.cwru.sepia.util.Direction;
import network.Network;
import network.layers.DenseLayer;
import network.math.Matrix;
import network.math.MyRand;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * An agent. Note that one agent controls multiple units
 */
public class Player implements Serializable {
    private Network brain; // Brain to make decisions

    private int fitness; // Fitness score

    private float color; // This player's "color" (used to see relations between players)

    float[] memory;

    public Player() {
        // Create random brain for this player
        brain = new Network(3); // 3 Memory neurons
        brain.addLayer(new DenseLayer(13, 16, "relu"));
        brain.addLayer(new DenseLayer(16, 16, "relu"));
        brain.addLayer(new DenseLayer(16, 11, "sigmoid"));
    }

    public Player(Network brain) {
       this.brain = brain;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * Reads data from the environment
     *
     * @return network.math.Matrix of environment observations to be fed to network
     */
    public Matrix observeEnvironment(State.StateView state, Integer unitID,
                                     List<Integer> myUnitIDs, List<Integer> enemyUnitIDs) {

        // Input data to neural network
        // Input consists of, in order:
        // This units's health. Then for each friendly unit:
        // Their relative x, y and health.
        // Then same for each enemy unit
        // Then 3 memory inputs from last time step
        // 1 + 3 * 1 + 3 * 2 + 3 = 13
        float[][] data = new float[1][13];

        // Where this unit is. network.Network sees other units relative to itself
        Unit.UnitView thisUnit = state.getUnit(unitID);
        int myHealth = thisUnit.getHP();
        int myX = thisUnit.getXPosition();
        int myY = thisUnit.getYPosition();

        // First input
        data[0][0] = myHealth;

        // Loop through remainder of friendly units
        int iUnit = 0; // Used to count units, as our doesn't count
        for (Integer id : myUnitIDs) {
            // Our unit isn't observed by itself
            if (!id.equals(unitID)) {
                // Collect data on unit
                Unit.UnitView unit = state.getUnit(id);
                UnitTemplate.UnitTemplateView unitTemplate = unit.getTemplateView();
                String unitName = unitTemplate.getName();
                int unitHealth = unit.getHP();
                int unitX = unit.getXPosition();
                int unitY = unit.getYPosition();
                int unitRange = unitTemplate.getRange();
                int unitDamage = unitTemplate.getBasicAttack();

                // Fill in inputs in right place
                if (unitX - myX == 0) {
                    data[0][3 * iUnit + 1] = 0.0f;

                } else {
                    data[0][3 * iUnit + 1] = 3.0f / (unitX - myX);
                }

                if (unitY - myY == 0) {
                    data[0][3 * iUnit + 2] = 0.0f;
                } else {
                    data[0][3 * iUnit + 2] = 3.0f / (unitY - myY);

                }
                data[0][3 * iUnit + 3] = unitHealth;

                iUnit++; // Next unit
//            data[0][i+3] = unitRange; // dont need, constant for now
//            data[0][i+4] = unitDamage; // same TODO add canMove?

//            System.out.printf("Unit: %s, Health: %d, X: %d, Y: %d, Range: %d, Damage: %d\n",
//                    unitName, unitHealth, unitX, unitY, unitRange, unitDamage);
            }
        }

        // Fill in all enemy input data
        for (int i = 0; i < enemyUnitIDs.size(); i++) {
            // Collect data on unit
            Unit.UnitView unit = state.getUnit(enemyUnitIDs.get(i));
            UnitTemplate.UnitTemplateView unitTemplate = unit.getTemplateView();
            String unitName = unitTemplate.getName();
            int unitHealth = unit.getHP();
            int unitX = unit.getXPosition();
            int unitY = unit.getYPosition();
            int unitRange = unitTemplate.getRange();
            int unitDamage = unitTemplate.getBasicAttack();

            // Our health + 3 data points per 1 friendly unit
            // means that enemy unit info starts at i=4
            data[0][3 * i + 4] = unitX - myX;
            data[0][3 * i + 5] = unitY - myY;
            data[0][3 * i + 6] = unitHealth;

//            System.out.printf("Enemy Unit: %s, Health: %d, X: %d, Y: %d, Range: %d, Damage: %d\n",
//                    unitName, unitHealth, unitX, unitY, unitRange, unitDamage);
        }

        for (int i = 0; i < 3; i++)
        {
//            System.out.println(brain.getMemory());
            data[0][10 + i] = brain.getMemory()[i];
        }

        // Create matrix with input data
        Matrix inputData = new Matrix(data);

        return inputData;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * Interprets the outputs of the neural network to different
     * actions and puts them in the actions object.
     *
     * @param data    output of network, in 1D array form
     * @param actions Resulting actions get put here
     */
    public void convertOutputToActions(float[] data, Map<Integer, Action> actions,
                                       Integer unitID, List<Integer> enemyUnitIDs) {
        // each network provides 8 output action slots:
        // move NE, SE, SW, NW, Attack, 3 slots for which unit to attack

        // First 5 determine movement direction, or attack
        // Take max, or no action if none > 0
        int maxIndex = -1;
        float maxValue = -1;
        for (int j = 0; j < 5; j++) {
            if (data[j] > 0.5 && data[j] > maxValue) {
                maxIndex = j;
                maxValue = data[j];
            }
        }

        // If the chosen index is one of the first 4,
        // Select a direction from the index
        if (maxIndex >= 0 && maxIndex < 4) {
            Direction dir = Direction.EAST; // Default value so it compiles
            switch (maxIndex) {
                case 0:
                    dir = Direction.NORTHEAST;
                    dir = Direction.NORTH;
                    break;
                case 1:
                    dir = Direction.SOUTHEAST;
                    dir = Direction.SOUTH;
                    break;
                case 2:
                    dir = Direction.SOUTHWEST;
                    dir = Direction.EAST;

                    break;
                case 3:
                    dir = Direction.NORTHWEST;
                    dir = Direction.WEST;
                    break;
                default:
                    System.err.println("Error: Bad movement index " + maxIndex);
                    break;
            }
            actions.put(unitID, Action.createPrimitiveMove(unitID, dir));
        }

        // This mean unit wants to attack
        if (maxIndex == 4) {
            // Find max value of indices 5-7,
            // this indicates which enemy to attack
            maxIndex = -1;
            maxValue = Integer.MIN_VALUE;
            for (int j = 5; j < 5 + enemyUnitIDs.size(); j++) {
                if (data[j] > maxValue) {
                    maxIndex = j - 5;
                    maxValue = data[j];
                }
            }

            // Look up the enemy id the network wanted to attack
            Integer enemyID = enemyUnitIDs.get(maxIndex);
            // TODO commented out for testing
//            actions.put(unitID, Action.createPrimitiveAttack(unitID, enemyID));
        }
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * Converts input data values to the range [-1, 1]
     * Using hardcoded ranges
     *
     * @param inputData The unstandardized data. Is modified in place
     */
    public void standardizeInputData(Matrix inputData) {
        // Hardcoded mean and std dev values
        // Choose so inputs are in rance (-1, 1)
        // These are: delX, delY, health for each unit
        float[] means = {0, 0, 25};
        float[] stds = {1, 1, 25};
        int nDataPerUnit = means.length;

        // Get reference to data
        float[] data = inputData.getData()[0];

        // Don't standardize three memory ones
        for (int i = 0; i < data.length - 3; i++) {
            // Index determines what type of data we are normalizing
            // We start with health for our unit, so it needs to be offset
            int index = (i + 2) % nDataPerUnit;
            // Standardize: todo: use multiplication not division for speed
            data[i] = (data[i] - means[index]) / stds[index];
        }
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * Uses information from the game state and last turn history
     * to find the member's fitness. Run every step
     *
     * @param state
     * @param history
     */
    public void middleFitnessUpdate(State.StateView state, History.HistoryView history,
                                    List<Integer> myUnitIDs, List<Integer> enemyUnitIDs) {
        // Current fitness function:
        // +1 point for every time step spent within 5 spaces of an enemy
        // This encourages agents to go near enemies (danger bonus)
        // 10 * damage dealt to opponents. This encourages agents to murder
        // And will overwhelm the danger bonus, so agents can ignore that
        // If fitness is 0 set it to 1 so the math doesn't break

        int dangerWeight = 1;
        int damageWeight = 20;
        int enemyDamageWeight = 0;
        int dangerDistance = 10;

        int turnNum = state.getTurnNumber();

        // Check danger bonus fitness
        // For every unit see if close to enemy
        for (Integer unitID : myUnitIDs) {
            // Get unit x and y
            Unit.UnitView unit = state.getUnit(unitID);
            int unitX = unit.getXPosition();
            int unitY = unit.getYPosition();

            // Check enemy units to see if we are close
            for (Integer enemyUnitID : enemyUnitIDs) {
                Unit.UnitView enemyUnit = state.getUnit(enemyUnitID);
                int enemyUnitX = enemyUnit.getXPosition();
                int enemyUnitY = enemyUnit.getYPosition();
                if (Math.max(Math.abs(enemyUnitX - unitX), Math.abs(enemyUnitY - unitY)) <= dangerDistance) {
                    fitness += 10 * 1.0 / Math.max(Math.abs(enemyUnitX - unitX), Math.abs(enemyUnitY - unitY));
//                    fitness += dangerWeight;
                    break; // Only one point per unit (not for each enemy)
                }
            }
        }

        // Check damage logs
        List<DamageLog> damageLogs = history.getDamageLogs(turnNum - 1);
        for (DamageLog damageLog : damageLogs) {
            // If we did the damage
            if (myUnitIDs.contains(damageLog.getAttackerID())) {
                // Add to fitness
                int damage = damageLog.getDamage();
                fitness += damageWeight * damage;
            } else  // The enemy attacked us
            {
                // Subtract from fitness
                int damage = damageLog.getDamage();
                fitness += enemyDamageWeight * damage;
            }
        }
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * Fitness calculations that are best done once, at the end of a run
     *
     * @param state World state
     * @param history World history
     * @param myUnitIDs Friendly unit ids
     * @param enemyUnitIDs Enemy unit ids
     */
    public void terminalFitnessUpdate(State.StateView state, History.HistoryView history,
                                      List<Integer> myUnitIDs, List<Integer> enemyUnitIDs) {

        // Make sure all fitnesses are at least non-0
        if (fitness <= 0) {
            fitness = 1;
        }
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * Uses the brain to decide upon actions
     * Which are processed into the game later
     * @param inputData Observed environment data
     * @return The brain's output data
     */
    public Matrix useBrain(Matrix inputData) {
        return brain.feedForward(inputData);
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    public Player crossover(Player partner){
        // New child with crossovered brain
        Player child = new Player(brain.crossover(partner.brain));

        // Crossover color randomly
        child.color = (MyRand.randFloat() > 0.5) ? this.color:partner.color;

        return child;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * Mutates this player
     * @param mutationRate How likely a mutation is
     * @param mutationSize Average size of a mutation
     */
    public void mutate(float mutationRate, float mutationSize)
    {
        brain.mutate(mutationRate, mutationSize);

        // Randomly mutate color
        if(MyRand.randFloat() < mutationRate){
            color += 0.25 * (MyRand.randFloat() - 0.5);
        }
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    /**
     * Clone player. Fitness is zero-ed out
     * This is not clone in the literal object sense
     * But in the reproduction sense
     * @return The cloned player
     */
    public Player clone(){
        Player p = new Player(brain.clone());
        p.color = color;
        return p;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

    public Network getBrain() {
        return brain;
    }

    public int getFitness() {
        return fitness;
    }

    public float getColor() {
        return color;
    }
    //---------------------------------------------------------------------------------------------------------------------------------------------------------

}
