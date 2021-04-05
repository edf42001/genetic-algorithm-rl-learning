package agents.interfaces;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.util.Direction;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class CrossEntropyAgentInterface implements AgentInterface {
    private int playernum;

    public CrossEntropyAgentInterface(int playernum)
    {
        this.playernum = playernum;
    }

    /**
     * Returns the observation state for a unit, in a form to be used with neural networks
     * @param unitID The unit observing
     * @param state Stateview
     * @return An array with all state variables
     */
    @Override
    public int[] observeUnitState(Integer unitID, State.StateView state, List<Integer> myUnitIDs, List<Integer> enemyUnitIDs) {
        // Total units, but don't count ourselves
//        int numUnits = myUnitIDs.size() + enemyUnitIDs.size() - 1;
        int numUnits = 3; // Always allocate space for 3 other units, but set their slots to 0 if they don't exist

        // x, y, distance, health, inRange
        int unitStateSize = 5;

        // For ourselves, we just see health
        int ourStateSize = 1;

        int stateSize = numUnits * unitStateSize + ourStateSize;
        int[] env = new int[stateSize];

        // Fill in default values of -1 million, so that dead units that don't fill in data can be differentiated
        // from a data value of 0
        Arrays.fill(env, -1000000);

        // Count where we are in the env array
        int index = 0;

        Unit.UnitView us = state.getUnit(unitID);
        int ourX = us.getXPosition();
        int ourY = us.getYPosition();
        int ourHealth = us.getHP();

        // Add our health
        env[index] = ourHealth;
        index = ourStateSize;

        // Loop through friendly units, get data
        for (Integer id : myUnitIDs)
        {
            // Don't count ourselves
            if (!id.equals(unitID))
            {
                Unit.UnitView unit = state.getUnit(id);
                int dx = unit.getXPosition() - ourX;
                int dy = ourY - unit.getYPosition(); // Flip y so that down is negative
                int health = unit.getHP();
                int dist = Math.abs(dx) + Math.abs(dy);
                // TODO Range of footmen is one, use UnitTemplateView to get range
                boolean inRange = Math.max(Math.abs(dx), Math.abs(dy)) <= 1;

                // Units expect to be player 1, in the left side of the field
                // Flip their state if they are on the other side of the field
                // so they get data they are used to
                if (playernum == 0)
                {
                    dx *= -1;
                    dy *= -1;
                }

                env[index] = dx;
                env[index+1] = dy;
                env[index+2] = dist;
                env[index+3] = health;
                env[index+4] = inRange ? 1 : 0;
            }
        }

        // 1 for our health, 1 friendly unit state size puts index here now
        index = 1 * unitStateSize + ourStateSize;

        // Exact same for enemies
        for (Integer id : enemyUnitIDs) {
            Unit.UnitView unit = state.getUnit(id);
            int dx = unit.getXPosition() - ourX;
            int dy = ourY - unit.getYPosition(); // Flip y so that down is negative
            int health = unit.getHP();
            int dist = Math.abs(dx) + Math.abs(dy);
            boolean inRange = Math.max(Math.abs(dx), Math.abs(dy)) <= 1;

            // Units expect to be player 1, in the left side of the field
            // Flip their state if they are on the other side of the field
            // so they get data they are used to
            if (playernum == 0)
            {
                dx *= -1;
                dy *= -1;
            }

            env[index] = dx;
            env[index + 1] = dy;
            env[index + 2] = dist;
            env[index + 3] = health;
            env[index + 4] = inRange ? 1 : 0;
            index += unitStateSize;
        }

        return env;
    }

    @Override
    public float getUnitLastReward(Integer unitID, State.StateView state, History.HistoryView history,
                                   List<Integer> myUnitIDs, List<Integer> enemyUnitIDs, boolean isFinalStep) {

        if (isFinalStep)
        {
            return getUnitFinalLastReward(unitID, state, history, myUnitIDs, enemyUnitIDs);
        }

        float stepReward = -0.03f;
        float distanceReward = 0.009f;
        float damageReward = 0.05f;
        float enemyDamageReward = -0.03f;

        float deathReward = 1.0f;

        float reward = 0;

        // Punishment for taking too long
        reward += stepReward;

        int turnNum = state.getTurnNumber();

        // Incentivize getting close to enemy
        // Reward falls off as 1 / distance to enemy
        Unit.UnitView unit = state.getUnit(unitID);
        Unit.UnitView enemy = state.getUnit(enemyUnitIDs.get(0));

        int unitX = unit.getXPosition();
        int unitY = unit.getYPosition();
        int enemyX = enemy.getXPosition();
        int enemyY = enemy.getYPosition();

        reward += distanceReward / (Math.abs(enemyX - unitX) + Math.abs(enemyY - unitY));

        // Reward agent for attacking enemy
        // Check damage logs
        List<DamageLog> damageLogs = history.getDamageLogs(turnNum - 1);
        for (DamageLog damageLog : damageLogs) {
            int damage = damageLog.getDamage();

            // If we did the damage
            if (unitID.equals(damageLog.getAttackerID())) {
                reward += damageReward * damage; // Add to reward
            }
            else if (unitID.equals(damageLog.getDefenderID())) // The enemy attacked us
            {
                reward += enemyDamageReward * damage; // Subtract from reward
            }
        }

        return reward;
    }

    public float getUnitFinalLastReward(Integer unitID, State.StateView state, History.HistoryView history,
                                        List<Integer> myUnitIDs, List<Integer> enemyUnitIDs)
    {
        float reward = 0;
        float stepReward = -0.03f;
        float damageReward = 0.05f;
        float enemyDamageReward = -0.03f;

        float winReward = 1.0f;
        float loseReward = 0.0f;

        // Punishment for taking too long
        reward += stepReward;

        // Reward for winning
        if (enemyUnitIDs.size() == 0)
        {
            reward += winReward;
        }

        if (myUnitIDs.size() == 0)
        {
            reward += loseReward;
        }

        int turnNum = state.getTurnNumber();

        // Reward agent for attacking enemy
        // Check damage logs
        List<DamageLog> damageLogs = history.getDamageLogs(turnNum - 1);
        for (DamageLog damageLog : damageLogs) {
            int damage = damageLog.getDamage();

            // If we did the damage
            if (unitID.equals(damageLog.getAttackerID())) {
                reward += damageReward * damage; // Add to reward
            }
            else if (unitID.equals(damageLog.getDefenderID())) // The enemy attacked us
            {
                reward += enemyDamageReward * damage; // Subtract from reward
            }
        }

        return reward;
    }

    @Override
    public void requestUnitAction(Integer unitID, int action, Map<Integer, Action> actions,
                                  State.StateView state, List<Integer> enemyUnitIDs) {
        if (action >= 0 && action < 4) {
            // Agents expect to be player 1, on the left of the screen
            // When an agent on the right of the screen says to go right,
            // they actually mean left. This flips left/right, up/down for that agent
            if (playernum == 0) {
                action = (action + 2) % 4;
            }

            Direction dir = Direction.NORTH; // Default value so it compiles
            switch (action) {
                case 0:
                    dir = Direction.NORTH;
                    break;
                case 1:
                    dir = Direction.EAST;
                    break;
                case 2:
                    dir = Direction.SOUTH;
                    break;
                case 3:
                    dir = Direction.WEST;
                    break;
                default:
                    System.err.println("Error: Bad action " + action);
                    break;
            }
            actions.put(unitID, Action.createPrimitiveMove(unitID, dir));
        } else if (action == 4) {
            Unit.UnitView unit = state.getUnit(unitID);

            // Find the closest enemy and attack them
            // may not be in range
            Integer closestEnemyID = enemyUnitIDs.get(0);
            int closestEnemyDist = Integer.MAX_VALUE;
            for (Integer enemyID : enemyUnitIDs) {
                Unit.UnitView enemy = state.getUnit(enemyID);

                int dx = enemy.getXPosition() - unit.getXPosition();
                int dy = enemy.getYPosition() - unit.getYPosition();

                if (Math.max(dx, dy) < closestEnemyDist) {
                    closestEnemyDist = Math.max(dx, dy);
                    closestEnemyID = enemyID;
                }
            }

            actions.put(unitID, Action.createPrimitiveAttack(unitID, closestEnemyID));
        } else if (action == -2) {
            System.err.println("Error: Something went wrong, bad action " + action);
        } else {
            System.out.println("Note: noop action " + action);
        }
    }
}
