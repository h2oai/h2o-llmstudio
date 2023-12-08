Feature: LLM Studio

    Background: LLM Studio user
        Given LLM Studio home page is opened
        When I login to LLM Studio
        Then I see the home page

#    Scenario: Import dataset using filesystem
#        When I upload dataset with path /home/llmstudio/mount/data/user/oasst/train_full.pq and name test_dataset
#        Then I should see the dataset test_dataset

#    Scenario: Delete dataset
#        When I delete dataset test_dataset
#        Then I should not see the dataset test_dataset

    Scenario: Create experiment
        When I create experiment test_experiment
        Then I should see the test_experiment should finish successfully 

