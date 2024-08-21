Feature: LLM Studio

    Background: LLM Studio user
        Given LLM Studio home page is opened
        When I login to LLM Studio
        Then I see the home page

    Scenario: Import dataset using filesystem
        When I upload dataset train_full.pq 
        And I name the dataset train-full.pq
        Then I should see the dataset train-full.pq
        When I delete dataset train-full.pq
        Then I should not see the dataset train-full.pq

    Scenario: Create experiment
        When I create experiment test-experiment
        And I update LLM Backbone to h2oai/llama2-0b-unit-test
        I set Mixed Precision to false
        And I tweak data sampling to 0.03
        And I tweak max length to 32
        And I select Perplexity metric
        And I run the experiment
        Then I should see the test-experiment should finish successfully 
        When I delete experiment test-experiment
        Then I should not see the experiment test-experiment
