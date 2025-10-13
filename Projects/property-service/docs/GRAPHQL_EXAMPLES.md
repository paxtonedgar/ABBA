# GraphQL Playground Test Queries

## 1) Owner: Create Property

```graphql
mutation CreateProperty {
  createProperty(input:{
    title:"Sunny Cabin"
    type: CABIN
    descriptionShort:"Cozy place with pines"
    descriptionLong:"A lovely 2BR/1BA cabin near the lake."
    maxGuests: 6
    bathroomsTotal: 1
    addressLine1:"100 Pine Way"
    city:"Evergreen"
    region:"CO"
    postalCode:"80439"
    country:"US"
    petsAllowed:true
    smokingAllowed:false
  }) {
    id
    title
    status
    city
    region
  }
}
```

**Note:** Keep the returned `id` as `PROPERTY_ID` for subsequent queries.

## 2) Owner: Upsert Rooms

```graphql
mutation UpsertRooms {
  upsertRooms(input:{
    propertyId:"PROPERTY_ID"
    rooms:[
      { name:"Bedroom 1", beds:[{type:QUEEN, count:1}] }
      { name:"Living Room", beds:[{type:SOFA_BED_FULL, count:1}] }
    ]
  }){
    id
    title
  }
}
```

## 3) Owner: Set Amenities

```graphql
mutation SetAmenities {
  setAmenities(input:{
    propertyId:"PROPERTY_ID"
    amenities: [
      { code:"wifi" }
      { code:"parking" }
      { code:"hot_tub" , details:"{\"private\":true}"}
    ]
  }){
    id
    title
  }
}
```

## 4) Owner: Submit for Review

```graphql
mutation SubmitProperty {
  submitProperty(input:{ propertyId:"PROPERTY_ID" }) {
    id
    status
    submittedAt
  }
}
```

## 5) Agent: Review Queue

```graphql
query AgentQueue {
  agentQueue {
    edges { node { id title status city region maxGuests } }
    totalCount
  }
}
```

## 6) Agent: Approve Property

```graphql
mutation ReviewApprove {
  reviewProperty(input:{ propertyId:"PROPERTY_ID", decision:APPROVE, notes:"Looks good" }) {
    id
    status
    reviewedAt
  }
}
```

## 7) Search with Filters

```graphql
query Search {
  properties(filter:{
    statusIn:[APPROVED]
    countryEq:"US"
    regionEq:"CO"
    minGuestsGte:4
    amenitiesAnyOf:["wifi","hot_tub"]
    fullText:"cabin"
  }) {
    edges { node { id title status city region maxGuests } }
    totalCount
  }
}
```

## 8) Get Amenities

```graphql
query GetAmenities {
  amenities {
    id
    code
    name
    category
  }
}
```

## 9) Get Rooms for Property

```graphql
query GetRooms {
  rooms(filter:{ propertyId:"PROPERTY_ID" }) {
    id
    name
    floorLabel
    areaSqFt
    ensuiteBath
    notes
  }
}
```

## 10) Get Property Details

```graphql
query GetProperty {
  property(id:"PROPERTY_ID") {
    id
    title
    status
    city
    region
    maxGuests
    bathroomsTotal
    petsAllowed
    smokingAllowed
    owner {
      id
      name
      email
    }
    rooms {
      id
      name
      floorLabel
      areaSqFt
    }
    amenities {
      id
      amenity {
        code
        name
        category
      }
    }
  }
}
```
