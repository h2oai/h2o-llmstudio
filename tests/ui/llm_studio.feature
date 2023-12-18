Feature: LLM Studio

    Background: LLM Studio user
        Given LLM Studio home page is opened
        When I login to LLM Studio
        Then I see the home page

    Scenario: Import dataset using filesystem
        When I upload dataset with path oasst/train_full.pq and name test-dataset
        Then I should see the dataset test-dataset

    Scenario: Delete dataset
        When I delete dataset test-dataset
        Then I should not see the dataset test-dataset

    Scenario: Create experiment
        When I create experiment test-experiment
        And I update LLM Backbone to MaxJeblick/llama2-0b-unit-test
        And I tweak data sampling to 0.1
        And I tweak max length prompt to 128
        And I tweak max length answer to 128
        And I tweak max length to 32
        And I run the experiment
        Then I should see the test-experiment should finish successfully 
        
    
    Scenario: Delete experiment
        When I delete experiment test-experiment
        Then I should not see the experiment test-experiment

