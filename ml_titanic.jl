### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 1d9a6e1b-a87d-4819-800e-846b3f73baee
using PlutoUI

# ╔═╡ 3aecd8a4-08f7-4bb3-b3e3-f5df68f98bae
using DataFrames

# ╔═╡ d1ed852f-ed46-4acf-a622-60e83bb9613a
using CSV

# ╔═╡ 2909d652-30ff-4f99-b6f4-a580f6013092
using StatPlots

# ╔═╡ e9b8263f-27e6-48aa-b973-2cdadc4db5c8
using Plots

# ╔═╡ 323f79cb-a539-482e-b268-7f6fc6cdb785
using Statistics

# ╔═╡ 48ef8873-2df0-4a3d-8a64-26c211b0ca0b
using DecisionTree

# ╔═╡ 00bb758d-db3d-4185-8f7f-e353469cd7e7
using ScikitLearn.CrossValidation: cross_val_score;

# ╔═╡ b0b44756-af9d-4bc2-9da2-90622677f907
using ScikitLearn

# ╔═╡ 6fa03634-fb92-11eb-20af-ff106bb525d4
md"""
# Titanic - Machine Learning from Disaster
"""

# ╔═╡ 0dcc1d5d-9cc5-4c46-87ee-f080d352f963
md"""I created this workbook to familiarize with Kaggle and ML with Julia"""

# ╔═╡ 66873f24-a310-4cb8-a9c7-747f28ca1667
PlutoUI.TableOfContents(aside=true, title="ML-Titanic ⛵⛵⛵")

# ╔═╡ 2cb98922-b13f-48fc-a347-8c24ba40ab22
md"""First we import the data from: https://www.kaggle.com/c/titanic/"""

# ╔═╡ 576c8e73-b6db-4e20-9d63-415e31d636e4
df_train = DataFrame(CSV.File("/Users/victor_sanz/downloads/train.csv"));

# ╔═╡ 40e1bf4e-741c-4ae3-a779-76cf74325b7d
df_test = DataFrame(CSV.File("/Users/victor_sanz/downloads/test.csv"));

# ╔═╡ 1eea202c-22ef-40c6-a67b-bbea1d8b71c2
df_test

# ╔═╡ 7e2c9a01-17e6-4f3a-8b6d-92069ff1e782
md"""### Exploratory Data Analysis"""

# ╔═╡ 6ab74fa1-7cdd-405d-96c4-5eb3fee5af00
md"""
Let's see how many men and women survived on average.
"""

# ╔═╡ 34d9ea51-44cc-47eb-a7a1-09a1f392c5c1
survived_by_sex = combine(groupby(df_train, :Sex), :Survived => length, :Survived => sum, :Survived => mean)

# ╔═╡ 506b8c4e-4acf-4f38-904a-d51b7b2102db
md"""
Now, let's see how many passengers survived by Pclass.
"""

# ╔═╡ 768721dc-42f1-4721-8661-f0d0692cb16a
survived_by_class = combine(groupby(df_train, :Pclass), :Survived => length, :Survived => sum, :Survived => mean)


# ╔═╡ 629bb806-1aac-4055-afe5-2c9e3cd3f248
combine_survivors = combine(groupby(df_train, [:Sex, :Pclass]), :Survived => length, :Survived => sum, :Survived => mean)

# ╔═╡ 7f6f04a5-acf9-4a60-8e68-1fd1e0670fb0
begin
y_val = combine_survivors.Survived_mean
nam = repeat("Class" .* string.(1:3), outer = 2)
sx = repeat(["Male", "Female"], inner = 3)
groupedbar(nam, y_val, group = sx, ylabel = "Mean", 
        title = "Means of survivors by class and gender")
end

# ╔═╡ 4aaea6ac-4d97-412b-bc7d-1309baad2cec
md"""### Missing Values"""

# ╔═╡ d0c07bec-72d5-4f48-9993-3aa9b6ba1636
md"""Let's fill the missing values. Age, Fare and Embarked by imputation"""

# ╔═╡ 3195e7c3-f9af-4d4f-916f-e333f16c1763
fill_missing = [:Age, :Fare, :Embarked]

# ╔═╡ 5b9dd8af-9442-4b79-b8c6-064b6d4b1db8
median_age = median(df_train.Age[.!ismissing.(df_train.Age)])

# ╔═╡ 286d058b-5f7a-4807-ae83-23f677b8854d
df_train.Age[ismissing.(df_train.Age)] .= median_age;

# ╔═╡ cedc1ec2-acac-4e97-993e-7ea4ece61583
missing_fare = ismissing.(df_train.Fare);

# ╔═╡ fd419396-655f-4d77-a452-bd7e6874ac5a
median_fare = median(df_train.Fare[.!missing_fare])

# ╔═╡ 9212a647-85f1-4d8f-9ac3-f8ce27e93ddb
df_train.Fare[missing_fare] .= median_fare;

# ╔═╡ be46f17c-6115-445f-877c-d46f693b2be8
combine(groupby(df_train, :Embarked), :Embarked => length)

# ╔═╡ 99f4b055-c2a7-4092-b3bb-26f63123f576
md"""C = Cherbourg, Q = Queenstown, S = Southampton, we'll sustitute the missing values by S, as it is the most common Port of Embarkation."""

# ╔═╡ dd3e04a3-c40e-4651-ba6a-8eac07e97d71
missings_emb = ismissing.(df_train.Embarked);

# ╔═╡ 47260f6b-ed55-4fa6-ad45-31e5932f15fd
df_train.Embarked[missings_emb] .= "S";

# ╔═╡ 1fa094d3-1a7a-4ab0-ae5c-ef52e657eb34
md"""### Categorical Variables"""

# ╔═╡ 4bc4a3b4-d4b3-4459-807a-dbc1822edba1
cat = [:Sex, :Embarked];

# ╔═╡ 889af367-88ee-4cc6-93df-cb25fe0f9dc8
begin
cod = Vector{Dict{String, Int64}}(undef, 2)
for (i, col) in enumerate(cat)
    cod[i] = Dict(val => num for (num, val) in enumerate(unique(df_train[:, col])))
end
end

# ╔═╡ aba92293-6449-4e1d-91eb-7090eb4072cc
md"""Now male => 1 and female => 2"""

# ╔═╡ 6b18fba3-3c0e-4e81-8489-e90f1576d60f
begin
df_train[:, :Sex_num] = [cod[1][x] for x in df_train.Sex]
df_train[1:10, [:Sex, :Sex_num]]
end

# ╔═╡ c97401af-7c7c-450f-bf54-231e0bbb09fb
md"""and S => 1, C => 2 , Q => 3"""

# ╔═╡ 11f82c17-b73d-4edb-820c-fde128bad79b
begin
df_train[:, :Embarked_num] = [cod[2][x] for x in df_train.Embarked]
df_train[1:10, [:Embarked, :Embarked_num]]	
end

# ╔═╡ 5475f876-eebe-43f5-a381-70a3245dc14e
md"""### Training Data"""

# ╔═╡ fdc345b2-9f20-44ac-9a3f-c8ab71bbe0b9
md"""
We discard Ticket and Cabin numbers.
"""

# ╔═╡ 7270e25b-c28b-493c-9b1e-52b4afae8864
var = [:Pclass, :Sex_num, :Age, :SibSp, :Parch, :Fare, :Embarked_num];

# ╔═╡ f5bc7202-3a82-4281-bbcb-520aad690380
begin
train_X = select(df_train, var)
train_y = df_train[!, :Survived]
end;

# ╔═╡ 9620b502-d799-4ddf-a9bc-62c074218ec5
train_X[1:10,1:7]

# ╔═╡ 9c4b3e53-21ff-43bc-8408-d8c7c64b7089
md"""#### Decission Tree Classifier"""

# ╔═╡ 1d9d11eb-5de3-4c6c-ac32-b4c666cef013
model1 = DecisionTreeClassifier(max_depth=100)

# ╔═╡ 43b00b1a-bc8a-4a8d-8568-a2fa059a3a66
train_matrix1 = Matrix{Float64}(train_X)

# ╔═╡ 4ebb92e8-d05d-4992-9a78-8ab458aa9384
fit!(model1, train_matrix1,train_y)

# ╔═╡ bf590abe-aa70-4856-9015-9e0c6934a488
cross_val_score(model1, train_matrix1, train_y;cv = 5)

# ╔═╡ 4802137b-c49f-49f4-b43e-09c51e2073ea


# ╔═╡ 94c4d313-4558-45e4-8a6b-7ddd8e43b1ec
md"""### Random Forest"""

# ╔═╡ 39e4cc2e-bd5a-413d-b597-85c9a67a475b
md"""We'll use a Random Forest"""

# ╔═╡ d513dc0f-ef62-4777-ba34-d7a8749c93fa
@sk_import ensemble: RandomForestClassifier;

# ╔═╡ 4e48bbe0-ea84-48bd-a6bc-be4d8d915c8a
model = RandomForestClassifier();

# ╔═╡ 510fd7a0-685a-4ff8-9ff7-3e2044fe3694
train_matrix = Matrix{Float64}(train_X);

# ╔═╡ b7b90b4d-86c1-4c44-971a-71de13264a2e
fit!(model, train_matrix, train_y)

# ╔═╡ b2b4f7e4-7a10-4aac-84ba-1d9ccd7a59d5
cross_val_score(model, train_matrix, train_y;cv = 5)

# ╔═╡ f2796382-a034-41e6-b6c4-8f176ffd7788
md"""### Preparing test"""

# ╔═╡ 21c0f1c4-c772-43d0-8722-302d999a5ec2
md"""We make the same changes as we did with the train data."""

# ╔═╡ 6e3048ed-4a2e-4a89-bc85-3fd24d4258f2
df_test.Age[ismissing.(df_test.Age)] .= median_age;

# ╔═╡ 456547d2-3855-4393-93e5-23b4bc102dc9
missing_test_fare = ismissing.(df_test.Fare);

# ╔═╡ a98803b6-b5f5-4f45-8c5d-61e9be012039
df_test.Fare[missing_test_fare] .= median_fare;

# ╔═╡ facade6b-8e72-4664-b849-da21a599fe2c
missing_test_emb = ismissing.(df_test.Embarked);

# ╔═╡ c09f6f04-6046-429e-8efd-5dd5e3f63e84
df_test.Embarked[missing_test_emb] .= "S";

# ╔═╡ 779e46cd-3f96-4e09-b968-c12ddfe6688b
df_test[:, :Sex_num] = [cod[1][x] for x in df_test.Sex];

# ╔═╡ deaed450-5dc9-4389-8e45-ab8abcc7daee
df_test[:, :Embarked_num] = [cod[2][x] for x in df_test.Embarked];

# ╔═╡ a117cf9d-bf11-49ad-8240-70f25ea6ea57
test_X = select(df_test, var)

# ╔═╡ b9e6432e-e6c5-4a83-826d-66a490c5e694
test_matrix = Matrix{Float64}(select(test_X, var));

# ╔═╡ dbb8def2-d8aa-4206-9a26-4b556fd0eaa1
md" ### Predictions "

# ╔═╡ 25f2c874-5fd1-40b0-b0a9-f4e7fd07fe63
md"""Predictions made by Decission Tree Classifier"""

# ╔═╡ 8f8047e6-d81b-4a33-93a8-1e68ae03b5ca
predictions_dtc = predict(model1, test_matrix)

# ╔═╡ 27838915-2844-4388-86b8-e055ab769d5a
md"""Predictions made by Random Forest Classifier"""

# ╔═╡ ed683c08-33e3-4c6d-9c45-06943f8f1a37
predictions_rfc = predict(model, test_matrix)

# ╔═╡ a1ed8892-e935-428b-ad68-3714b26339b1
length(predictions_rfc)

# ╔═╡ 6afd57df-2b3d-4c72-9fb7-4f5999a993b6
pass_id = df_test[!,"PassengerId"]

# ╔═╡ 5420a352-e157-4f23-90f8-f516cb94d8a3
output1 = DataFrame( PassengerId = pass_id,Survived_dtc =predictions_dtc);

# ╔═╡ 2a13c07f-0e8b-4367-944b-37eefa3366e3
output2 = DataFrame( PassengerId = pass_id,Survived =predictions_rfc);

# ╔═╡ ee4a85e7-932d-496a-a122-5f966dfc261a
begin
surv_dtc=sum(output1[!,2])
surv_rfc=sum(output2[!,2])
md"Decision Tree Classifier gave us $surv_dtc survivors for the test data while Random Forest Classfier gave us $surv_rfc survivors."
end

# ╔═╡ 7b49a1db-941e-46f5-b878-ce1237a00413
md""" ### Final comments"""

# ╔═╡ 908cbde7-0178-4e00-b41f-0eb7ca9e8f07
begin
df_test_sex = df_test[!,4]
df_test_class = df_test[!,2]
df_test_id = df_test[!,1]
end;

# ╔═╡ d79a7fd1-d788-46a8-aafc-26c09a3e4c05
df_test_survivors = DataFrame(PassengerId=df_test_id, Pclass = df_test_class,Sex = df_test_sex,Survived =predictions_rfc)

# ╔═╡ 6cbfa298-e804-4047-ad23-05a0695a9c1c
survived_by_sext = combine(groupby(df_test_survivors, :Sex), :Survived => length, :Survived => sum, :Survived => mean)

# ╔═╡ 55dfeb8b-c8dc-46ad-a007-864a1313e224
survived_by_classt = combine(groupby(df_test_survivors, :Pclass), :Survived => length, :Survived => sum, :Survived => mean)

# ╔═╡ dd704a31-f5a4-4b08-bd1d-50e0fc867fd8
combine_survivorst = combine(groupby(df_test_survivors, [:Sex, :Pclass]), :Survived => length, :Survived => sum, :Survived => mean)

# ╔═╡ 5cb3b916-7eae-4223-9f53-5f7ceb7eefff
begin
y_valt = combine_survivorst.Survived_mean
namt = repeat("Class" .* string.(1:3), outer = 2)
sxt = repeat(["Male", "Female"], inner = 3)
groupedbar(nam, y_val, group = sx, ylabel = "Mean", 
        title = "Means of survivors by class and gender")
end

# ╔═╡ 5fb0e787-b15e-476a-8ccf-01c8631370d3
md"""Finally we export our results to a csv file:"""

# ╔═╡ ccb3e06c-d7fc-41fd-9b22-27427f986477
CSV.write("/Users/victor_sanz/downloads/my_submission.csv", output2)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DecisionTree = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ScikitLearn = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
StatPlots = "60ddc479-9b66-56df-82fc-76a74619b69c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
CSV = "~0.8.5"
DataFrames = "~1.2.2"
DecisionTree = "~0.10.10"
Plots = "~0.29.9"
PlutoUI = "~0.7.1"
ScikitLearn = "~0.6.4"
StatPlots = "~0.9.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[Artifacts]]
deps = ["Pkg"]
git-tree-sha1 = "c30985d8821e0cd73870b17b0ed0ce6dc44cb744"
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.3.0"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c3598e525718abcc440f69cc6d5f60dda0a1b61e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.6+5"

[[CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bdc0937269321858ab2a4f288486cb258b9a0af7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.0"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "32a2b8af383f11cbb65803883837a149d10dfe8a"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.10.12"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "344f143fa0ec67e47917848795ab19c6a455f32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.32.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8e695f735fca77e9708e795eda62afdb869cbb70"
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.3.4+0"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DecisionTree]]
deps = ["DelimitedFiles", "Distributed", "LinearAlgebra", "Random", "ScikitLearnBase", "Statistics", "Test"]
git-tree-sha1 = "8b58db7954a6206399d9f66ef1a328da8c0f1d19"
uuid = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
version = "0.10.10"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "abe4ad222b26af3337262b8afb28fab8d215e9f8"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.3"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3676697fd903ba314aaaa0ec8d6813b354edb875"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.23.11"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "c82bef6fc01e30d500f588cd01d29bdd44f1924e"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.3.0"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "LibVPX_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3cc57ad0a213808473eafef4845a74766242e05f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.3.1+4"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "IntelOpenMP_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Reexport"]
git-tree-sha1 = "1b48dbde42f307e48685fa9213d8b9f8c0d87594"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.3.2"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3676abafff7e4ff07bbd2c42b3d8201f31653dcc"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.9+8"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "502b3de6039d5b78c76118423858d981349f3823"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.9.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "cbd58c9deb1d304f5a245a0b7eb841a2560cfec6"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.1+5"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0d20aed5b14dd4c9a2453c1b601d08e1149679cc"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.5+6"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GR]]
deps = ["Base64", "DelimitedFiles", "LinearAlgebra", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "7ea6f715b7caa10d7ee16f1cfcd12f3ccc74116a"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.48.0"

[[GeometryTypes]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "07194161fe4e181c6bf51ef2e329ec4e7d050fc4"
uuid = "4d00f742-c7ba-57c2-abde-4428a4b178cb"
version = "0.8.4"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IterableTables]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Requires", "TableTraits", "TableTraitsUtils"]
git-tree-sha1 = "70300b876b2cebde43ebc0df42bc8c94a144e1b4"
uuid = "1c8ee90f-4401-5389-894e-7a04a3dc0f4d"
version = "1.0.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "df381151e871f41ee86cee4f5f6fd598b8a68826"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.0+3"

[[LazyArtifacts]]
deps = ["Pkg"]
git-tree-sha1 = "4bb5499a1fc437342ea9ab7e319ede5a457c0968"
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.3.0"

[[LibGit2]]
deps = ["Printf"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibVPX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "85fcc80c3052be96619affa2fe2e6d2da3908e11"
uuid = "dd192d2f-8180-539f-9fb4-cc70b1dcf69a"
version = "1.9.0+1"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "c253236b0ed414624b083e6b72bfe891fbd2c7af"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+1"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f8c673ccc215eb50fcadb285f522420e29e69e1c"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "0.4.5"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[Observables]]
git-tree-sha1 = "3469ef96607a6b9a1e89e54e6f23401073ed3126"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.3.3"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0f4a4836e5f3e0763243b8324200af6d0e0f90c"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.5"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "a42c0f138b9ebe8b58eba2271c5053773bde52d0"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.4+2"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "71bbbc616a1d710879f5a1021bcba65ffba6ce58"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.1+6"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9db77584158d0ab52307f8c04f8e7c08ca76b5b3"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.3+4"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f9d57f4126c39565e05a2b0264df99f497fc6f37"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.1+3"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "95a4038d1011dfdbde7cecd2ad0ac411e53ab1bc"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.10.1"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "2276ac65f1e236e0a6ea70baff3f62ad4c625345"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.2"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "bfd7d8c7fd87f04543810d9cbd3995972236ba1b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.2"

[[Pkg]]
deps = ["Dates", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "UUIDs"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "87a4ea7f8c350d87d3a8ca9052663b633c0b2722"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "1.0.3"

[[PlotUtils]]
deps = ["Colors", "Dates", "Printf", "Random", "Reexport"]
git-tree-sha1 = "51e742162c97d35f714f9611619db6975e19384b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "0.6.5"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "FFMPEG", "FixedPointNumbers", "GR", "GeometryTypes", "JSON", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "Reexport", "Requires", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "f226ff9b8e391f6a10891563c370aae8beb5d792"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "0.29.9"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "Logging", "Markdown", "Random", "Suppressor"]
git-tree-sha1 = "45ce174d36d3931cd4e37a47f93e07d1455f038d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.1"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "cde4ce9d6f33219465b55162811d8de8139c0414"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.2.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "169bb8ea6b1b143c5cf57df6d34d022a7b60c6db"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.3"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "7dff99fbc740e2f8228c6878e2aad6d7c2678098"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.1"

[[RecipesBase]]
git-tree-sha1 = "b4ed4a7f988ea2340017916f7c9e5d7560b52cae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "0.8.0"

[[Reexport]]
deps = ["Pkg"]
git-tree-sha1 = "7b1d07f411bc8ddb7977ec7f377b97b158514fe0"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "0.2.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "86c5647b565873641538d8f812c04e4c9dbeb370"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.6.1"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "1b7bf41258f6c5c9c31df8c1ba34c1fc88674957"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.2.2+2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[ScikitLearn]]
deps = ["Compat", "Conda", "DataFrames", "Distributed", "IterTools", "LinearAlgebra", "MacroTools", "Parameters", "Printf", "PyCall", "Random", "ScikitLearnBase", "SparseArrays", "StatsBase", "VersionParsing"]
git-tree-sha1 = "ccb822ff4222fcf6ff43bbdbd7b80332690f168e"
uuid = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
version = "0.6.4"

[[ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "a3a337914a035b2d59c9cbe7f1a38aaba1265b02"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.6"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "ee010d8f103468309b8afac4abb9be2e18ff1182"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "0.3.2"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["OpenSpecFun_jll"]
git-tree-sha1 = "d8d8b8a9f4119829410ecd706da4cc8594a1e020"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "0.10.3"

[[StatPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "IterableTables", "KernelDensity", "Observables", "Plots", "RecipesBase", "Reexport", "StatsBase", "TableTraits", "TableTraitsUtils", "Test", "Widgets"]
git-tree-sha1 = "245c50f8a6534bb16ada031e064363f8298b61b9"
uuid = "60ddc479-9b66-56df-82fc-76a74619b69c"
version = "0.9.2"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "fed1ec1e65749c4d96fc20dd13bea72b55457e62"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.9"

[[StatsFuns]]
deps = ["LogExpFunctions", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "30cd8c360c54081f806b1ee14d2eecbef3c04c49"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.8"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
git-tree-sha1 = "44aaac2d2aec4a850302f9aa69127c74f0c3787e"
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[TableTraitsUtils]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Missings", "TableTraits"]
git-tree-sha1 = "8fc12ae66deac83e44454e61b02c37b326493233"
uuid = "382cd787-c1b6-5bf2-a167-d5b971a19bda"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

[[Test]]
deps = ["Distributed", "InteractiveUtils", "Logging", "Random"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "eae2fbbc34a79ffd57fb4c972b08ce50b8f6a00d"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.3"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

[[Zlib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "320228915c8debb12cb434c59057290f0834dbf6"
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.11+18"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "acc685bcf777b2202a904cdcb49ad34c2fa1880c"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.14.0+4"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7a5780a0d9c6864184b3a2eeeb833a0c871f00ab"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "0.1.6+4"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "fa14ac25af7a4b8a7f61b287a124df7aab601bcd"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.6+6"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d713c1ce4deac133e3334ee12f4adff07f81778f"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2020.7.14+2"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "487da2f8f2f0c8ee0e83f39d13037d6bbf0a45ab"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.0.0+3"
"""

# ╔═╡ Cell order:
# ╟─6fa03634-fb92-11eb-20af-ff106bb525d4
# ╟─0dcc1d5d-9cc5-4c46-87ee-f080d352f963
# ╠═1d9a6e1b-a87d-4819-800e-846b3f73baee
# ╠═66873f24-a310-4cb8-a9c7-747f28ca1667
# ╠═3aecd8a4-08f7-4bb3-b3e3-f5df68f98bae
# ╠═d1ed852f-ed46-4acf-a622-60e83bb9613a
# ╠═2909d652-30ff-4f99-b6f4-a580f6013092
# ╟─2cb98922-b13f-48fc-a347-8c24ba40ab22
# ╠═576c8e73-b6db-4e20-9d63-415e31d636e4
# ╠═1eea202c-22ef-40c6-a67b-bbea1d8b71c2
# ╠═40e1bf4e-741c-4ae3-a779-76cf74325b7d
# ╟─7e2c9a01-17e6-4f3a-8b6d-92069ff1e782
# ╟─6ab74fa1-7cdd-405d-96c4-5eb3fee5af00
# ╠═e9b8263f-27e6-48aa-b973-2cdadc4db5c8
# ╠═323f79cb-a539-482e-b268-7f6fc6cdb785
# ╠═34d9ea51-44cc-47eb-a7a1-09a1f392c5c1
# ╟─506b8c4e-4acf-4f38-904a-d51b7b2102db
# ╠═768721dc-42f1-4721-8661-f0d0692cb16a
# ╠═629bb806-1aac-4055-afe5-2c9e3cd3f248
# ╟─7f6f04a5-acf9-4a60-8e68-1fd1e0670fb0
# ╟─4aaea6ac-4d97-412b-bc7d-1309baad2cec
# ╟─d0c07bec-72d5-4f48-9993-3aa9b6ba1636
# ╠═3195e7c3-f9af-4d4f-916f-e333f16c1763
# ╠═5b9dd8af-9442-4b79-b8c6-064b6d4b1db8
# ╠═286d058b-5f7a-4807-ae83-23f677b8854d
# ╠═cedc1ec2-acac-4e97-993e-7ea4ece61583
# ╠═fd419396-655f-4d77-a452-bd7e6874ac5a
# ╠═9212a647-85f1-4d8f-9ac3-f8ce27e93ddb
# ╠═be46f17c-6115-445f-877c-d46f693b2be8
# ╟─99f4b055-c2a7-4092-b3bb-26f63123f576
# ╠═dd3e04a3-c40e-4651-ba6a-8eac07e97d71
# ╠═47260f6b-ed55-4fa6-ad45-31e5932f15fd
# ╟─1fa094d3-1a7a-4ab0-ae5c-ef52e657eb34
# ╠═4bc4a3b4-d4b3-4459-807a-dbc1822edba1
# ╠═889af367-88ee-4cc6-93df-cb25fe0f9dc8
# ╟─aba92293-6449-4e1d-91eb-7090eb4072cc
# ╟─6b18fba3-3c0e-4e81-8489-e90f1576d60f
# ╟─c97401af-7c7c-450f-bf54-231e0bbb09fb
# ╟─11f82c17-b73d-4edb-820c-fde128bad79b
# ╟─5475f876-eebe-43f5-a381-70a3245dc14e
# ╟─fdc345b2-9f20-44ac-9a3f-c8ab71bbe0b9
# ╠═7270e25b-c28b-493c-9b1e-52b4afae8864
# ╠═f5bc7202-3a82-4281-bbcb-520aad690380
# ╟─9620b502-d799-4ddf-a9bc-62c074218ec5
# ╟─9c4b3e53-21ff-43bc-8408-d8c7c64b7089
# ╠═48ef8873-2df0-4a3d-8a64-26c211b0ca0b
# ╠═1d9d11eb-5de3-4c6c-ac32-b4c666cef013
# ╠═43b00b1a-bc8a-4a8d-8568-a2fa059a3a66
# ╠═4ebb92e8-d05d-4992-9a78-8ab458aa9384
# ╠═00bb758d-db3d-4185-8f7f-e353469cd7e7
# ╠═bf590abe-aa70-4856-9015-9e0c6934a488
# ╠═4802137b-c49f-49f4-b43e-09c51e2073ea
# ╟─94c4d313-4558-45e4-8a6b-7ddd8e43b1ec
# ╟─39e4cc2e-bd5a-413d-b597-85c9a67a475b
# ╠═d513dc0f-ef62-4777-ba34-d7a8749c93fa
# ╠═b0b44756-af9d-4bc2-9da2-90622677f907
# ╠═4e48bbe0-ea84-48bd-a6bc-be4d8d915c8a
# ╠═510fd7a0-685a-4ff8-9ff7-3e2044fe3694
# ╠═b7b90b4d-86c1-4c44-971a-71de13264a2e
# ╠═b2b4f7e4-7a10-4aac-84ba-1d9ccd7a59d5
# ╟─f2796382-a034-41e6-b6c4-8f176ffd7788
# ╟─21c0f1c4-c772-43d0-8722-302d999a5ec2
# ╠═6e3048ed-4a2e-4a89-bc85-3fd24d4258f2
# ╠═456547d2-3855-4393-93e5-23b4bc102dc9
# ╠═a98803b6-b5f5-4f45-8c5d-61e9be012039
# ╠═facade6b-8e72-4664-b849-da21a599fe2c
# ╠═c09f6f04-6046-429e-8efd-5dd5e3f63e84
# ╠═779e46cd-3f96-4e09-b968-c12ddfe6688b
# ╠═deaed450-5dc9-4389-8e45-ab8abcc7daee
# ╠═a117cf9d-bf11-49ad-8240-70f25ea6ea57
# ╠═b9e6432e-e6c5-4a83-826d-66a490c5e694
# ╟─dbb8def2-d8aa-4206-9a26-4b556fd0eaa1
# ╠═25f2c874-5fd1-40b0-b0a9-f4e7fd07fe63
# ╠═8f8047e6-d81b-4a33-93a8-1e68ae03b5ca
# ╟─27838915-2844-4388-86b8-e055ab769d5a
# ╠═ed683c08-33e3-4c6d-9c45-06943f8f1a37
# ╠═a1ed8892-e935-428b-ad68-3714b26339b1
# ╠═6afd57df-2b3d-4c72-9fb7-4f5999a993b6
# ╠═5420a352-e157-4f23-90f8-f516cb94d8a3
# ╠═2a13c07f-0e8b-4367-944b-37eefa3366e3
# ╟─ee4a85e7-932d-496a-a122-5f966dfc261a
# ╟─7b49a1db-941e-46f5-b878-ce1237a00413
# ╠═908cbde7-0178-4e00-b41f-0eb7ca9e8f07
# ╠═d79a7fd1-d788-46a8-aafc-26c09a3e4c05
# ╠═6cbfa298-e804-4047-ad23-05a0695a9c1c
# ╠═55dfeb8b-c8dc-46ad-a007-864a1313e224
# ╠═dd704a31-f5a4-4b08-bd1d-50e0fc867fd8
# ╟─5cb3b916-7eae-4223-9f53-5f7ceb7eefff
# ╟─5fb0e787-b15e-476a-8ccf-01c8631370d3
# ╠═ccb3e06c-d7fc-41fd-9b22-27427f986477
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
