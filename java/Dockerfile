FROM maven:3-amazoncorretto-8 as build

# Copy specific files. Could probably just use .dockerignore
WORKDIR /usr/src/myapp
COPY data/ReinforcementLearning.xml data/ReinforcementLearningConfig.xml data/
COPY enemy_agents enemy_agents/
COPY src src/
COPY lib lib/
COPY pom.xml .

# Compile java
RUN mvn compile && \
    # Needed because I don't know if maven can have dependencies on .class so move it into the classpath
    mv enemy_agents/CombatAgentEnemy.class target/classes

ENTRYPOINT mvn exec:java -Dexec.mainClass="edu.cwru.sepia.Main2" -Dexec.classpathScope=compile \
           -Dexec.args="data/ReinforcementLearningConfig.xml" >> rl_learning_java_stdout.txt

# Plan to copy only executable files. Doesn't seem necessary right now but might reduce image size?
# FROM openjdk:8
# COPY --from=build /usr/src/myapp/java/target/mavenGrpcTest-1.0-SNAPSHOT.jar /usr/src/myapp/java/mavenGrpcTest-1.0-SNAPSHOT.jar
# WORKDIR /usr/src/myapp/java

# To Run:
#docker run --rm -e AWS_REGION="us-east-1" -e S3_PATH="rl-training-run-results" rl:latest
