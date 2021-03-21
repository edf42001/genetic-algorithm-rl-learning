/**
 *  Strategy Engine for Programming Intelligent Agents (SEPIA)
 Copyright (C) 2012 Case Western Reserve University
 This file is part of SEPIA.
 SEPIA is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 SEPIA is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with SEPIA.  If not, see <http://www.gnu.org/licenses/>.
 */
package runners;

import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.StateCreator;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.XmlStateCreator;
import network.math.MyRand;

import java.io.IOException;

public class RandomStateCreator implements StateCreator {
    private static final long serialVersionUID = 1L;

    private final XmlStateCreator stateCreator;

    public RandomStateCreator(XmlStateCreator stateCreator) {
        this.stateCreator = stateCreator;
    }

    @Override
    public State createState() {
        try
        {
            State state = stateCreator.createState();
            randomizeUnitLocations(state, 0);
            randomizeUnitLocations(state, 1);
            return state;
        }
        catch(Exception ex)
        {
            ex.printStackTrace();
            return null;
        }
    }

    public void randomizeUnitLocations(State state, int playernum)
    {
        int xCenter = 4;
        int xWidth = 6;
        int yCenter = state.getYExtent() / 2;
        int yWidth = (int) (0.4 * state.getYExtent());

        for (Integer unitID : state.getUnits(playernum).keySet())
        {
            int x = -1;
            int y = -1;
            do {
                x = MyRand.randInt(xWidth) + xCenter - xWidth / 2;
                y = MyRand.randInt(yWidth) + yCenter - yWidth / 2;

                // Units expect to be player 1, on the left
                // Flip this calculation so other units spawn on the right
                if (playernum == 0)
                {
                    y = state.getYExtent() - y - 1;
                    x = state.getXExtent() - x - 1;
                }
            } while (locationOccupied(state, unitID, x, y));

            state.getUnit(unitID).setxPosition(x);
            state.getUnit(unitID).setyPosition(y);
        }
    }

    public boolean locationOccupied(State state, Integer unitID, int x, int y)
    {
        for (Integer id : state.getUnits().keySet())
        {
            Unit unit = state.getUnit(id);
            int unitX = unit.getxPosition();
            int unitY = unit.getyPosition();

            if (!id.equals(unitID) && x == unitX && y == unitY)
            {
                return true;
            }
        }
        return false;
    }

}
